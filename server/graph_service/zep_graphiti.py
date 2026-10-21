import logging
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # type: ignore
from graphiti_core.utils.datetime_utils import utc_now  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        **kwargs,
    ):
        super().__init__(uri, user, password, llm_client, **kwargs)  # type: ignore

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def ensure_episode(
        self,
        uuid: str,
        group_id: str,
        name: str,
        content: str,
        reference_time: datetime,
        source_description: str,
    ) -> bool:
        """Create the episode with a client-chosen uuid if it does not exist yet.

        Returns True when the episode was created here, False when it already existed.

        graphiti_core's add_episode(uuid=...) only re-processes an existing episode, so a new
        message that carries its own uuid would fail with NodeNotFoundError.
        """
        try:
            await EpisodicNode.get_by_uuid(self.driver, uuid)
            return False
        except NodeNotFoundError:
            pass
        await EpisodicNode(
            uuid=uuid,
            name=name,
            group_id=group_id,
            labels=[],
            source=EpisodeType.message,
            content=content,
            source_description=source_description,
            created_at=utc_now(),
            valid_at=reference_time,
        ).save(self.driver)
        return True

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _create_ai_clients(settings: ZepEnvDep) -> dict:
    """Build LLM, embedder and reranker clients for an OpenAI-compatible provider.

    Returns an empty dict (graphiti defaults, i.e. OpenAI) when no base URL is configured.
    """
    llm_base_url = settings.llm_base_url or settings.openai_base_url
    embedder_base_url = settings.embedder_base_url or settings.openai_base_url
    clients: dict = {}

    def _openai(base_url: str):
        # Bounded timeouts: the SDK default (600 s read, 2 retries) let one stalled provider
        # request hold the single ingest worker for up to ~30 minutes.
        import httpx
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=base_url,
            timeout=httpx.Timeout(settings.llm_timeout, connect=10.0),
            max_retries=settings.llm_max_retries,
        )

    if llm_base_url:
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        model = settings.model_name or 'gpt-4.1-mini'
        clients['llm_client'] = OpenAIGenericClient(
            config=LLMConfig(
                api_key=settings.openai_api_key,
                base_url=llm_base_url,
                model=model,
                small_model=settings.small_model_name or model,
            ),
            client=_openai(llm_base_url),
            max_tokens=settings.llm_max_tokens,
            structured_output_mode=settings.llm_structured_output_mode,  # type: ignore[arg-type]
        )
        # The reranker needs logprobs + logit_bias, so pick a model that supports them.
        clients['cross_encoder'] = OpenAIRerankerClient(
            config=LLMConfig(
                api_key=settings.openai_api_key,
                base_url=llm_base_url,
                model=settings.reranker_model_name or 'gpt-4o-mini',
            ),
            client=_openai(llm_base_url),
        )

    if embedder_base_url:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        clients['embedder'] = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=settings.openai_api_key,
                base_url=embedder_base_url,
                embedding_model=settings.embedding_model_name or 'text-embedding-3-small',
            ),
            client=_openai(embedder_base_url),
        )

    return clients


def _create_graphiti_client(settings: ZepEnvDep) -> ZepGraphiti:
    """Create a ZepGraphiti client based on the configured database backend."""
    if settings.db_backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver_cls = FalkorDriver
        if settings.falkordb_single_graph:

            class SingleGraphFalkorDriver(FalkorDriver):
                # graphiti_core puts every group_id in its own FalkorDB graph, but the server's
                # by-uuid endpoints (GET/DELETE /entity-edge, DELETE /episode, DELETE /group)
                # only look in the default graph. Keep all groups in FALKORDB_DATABASE and rely
                # on the group_id property instead, as with Neo4j.
                def clone(self, database: str):  # type: ignore[override]
                    return self

            driver_cls = SingleGraphFalkorDriver

        driver = driver_cls(  # type: ignore
            host=settings.falkordb_host or 'localhost',  # type: ignore
            port=settings.falkordb_port or 6379,  # type: ignore
            database=settings.falkordb_database or 'default_db',  # type: ignore
            username=settings.falkordb_username or None,
            password=settings.falkordb_password or None,
        )
        return ZepGraphiti(graph_driver=driver, **_create_ai_clients(settings))  # type: ignore
    else:
        # Validate Neo4j settings are present
        if not all([settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password]):
            raise ValueError(
                'Neo4j configuration (neo4j_uri, neo4j_user, neo4j_password) is required '
                "when db_backend is 'neo4j'"
            )
        return ZepGraphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            **_create_ai_clients(settings),
        )


def create_graphiti(settings) -> ZepGraphiti:
    client = _create_graphiti_client(settings)
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name
    return client


async def get_graphiti(settings: ZepEnvDep):
    client = create_graphiti(settings)
    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    try:
        await client.build_indices_and_constraints()
    finally:
        await client.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        episodes=edge.episodes or [],
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
