from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    # OpenAI-compatible providers (e.g. OpenRouter). When an LLM base URL is set
    # (LLM_BASE_URL, falling back to OPENAI_BASE_URL) the LLM uses OpenAIGenericClient
    # (chat.completions + json_schema) and the reranker uses the same endpoint.
    # The embedder uses EMBEDDER_BASE_URL, falling back to OPENAI_BASE_URL.
    llm_base_url: str | None = Field(None)
    embedder_base_url: str | None = Field(None)
    small_model_name: str | None = Field(None)
    reranker_model_name: str | None = Field(None)
    # 'json_schema' (default) or 'json_object' (schema in the prompt). OpenRouter's OpenAI
    # route enforces strict schemas, which graphiti's models don't meet: use 'json_object'.
    llm_structured_output_mode: str = Field('json_schema')
    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None)
    falkordb_host: str | None = Field(None)
    falkordb_port: int | None = Field(None)
    falkordb_database: str | None = Field(None)
    falkordb_username: str | None = Field(None)
    falkordb_password: str | None = Field(None)
    # One FalkorDB graph for all group_ids (see zep_graphiti.py). Set false for upstream's graph-per-group.
    falkordb_single_graph: bool = Field(True)
    db_backend: str = Field('neo4j')
    # Background ingest (/messages). A job that exceeds INGEST_JOB_TIMEOUT seconds is cancelled,
    # logged and dropped so it cannot block the queue; INGEST_CONCURRENCY workers run in parallel.
    ingest_job_timeout: float = Field(180.0)
    ingest_concurrency: int = Field(1, ge=1, le=8)
    # Per-HTTP-request timeout (seconds) and SDK retries for the LLM, embedder and reranker clients
    # when an OpenAI-compatible base URL is set. The OpenAI SDK default is 600 s x 3 attempts.
    llm_timeout: float = Field(60.0)
    llm_max_retries: int = Field(1, ge=0)
    # Upper bound on completion tokens per LLM call (OpenAIGenericClient defaults to 16384).
    llm_max_tokens: int = Field(8192)

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
