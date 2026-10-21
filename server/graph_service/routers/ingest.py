import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.config import get_settings
from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep, create_graphiti

logger = logging.getLogger('graph_service.ingest')


@dataclass
class Job:
    id: int
    label: str
    run: Callable[[ZepGraphiti], Awaitable[None]]
    on_failure: Callable[[ZepGraphiti], Awaitable[None]] | None = None
    enqueued_at: float = 0.0


class AsyncWorker:
    """Background queue for /messages.

    Jobs run on a long-lived Graphiti client owned by the worker (the request-scoped client is
    closed as soon as the 202 response is sent). Each job is bounded by INGEST_JOB_TIMEOUT and
    every outcome (start / done / failed / timeout) is logged, so a stuck provider call can no
    longer stall the whole queue silently.
    """

    def __init__(self):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.graphiti: ZepGraphiti | None = None
        self.timeout = 180.0
        self.in_progress = 0
        self.stats = {'done': 0, 'failed': 0, 'timeout': 0}
        self._ids = itertools.count(1)

    def next_id(self) -> int:
        return next(self._ids)

    async def put(self, job: Job):
        job.enqueued_at = time.monotonic()
        await self.queue.put(job)
        logger.info('ingest queued job=%d %s queue=%d', job.id, job.label, self.queue.qsize())

    async def _run_one(self, worker_id: int, job: Job):
        assert self.graphiti is not None
        waited = time.monotonic() - job.enqueued_at
        logger.info(
            'ingest start job=%d worker=%d %s waited=%.1fs queue=%d',
            job.id,
            worker_id,
            job.label,
            waited,
            self.queue.qsize(),
        )
        t0 = time.monotonic()
        self.in_progress += 1
        try:
            await asyncio.wait_for(job.run(self.graphiti), timeout=self.timeout)
            self.stats['done'] += 1
            logger.info('ingest done job=%d %s in %.1fs', job.id, job.label, time.monotonic() - t0)
            return
        except asyncio.TimeoutError:
            self.stats['timeout'] += 1
            logger.error(
                'ingest timeout job=%d %s after %.1fs (limit %.0fs)',
                job.id,
                job.label,
                time.monotonic() - t0,
                self.timeout,
            )
        except Exception as e:
            self.stats['failed'] += 1
            logger.exception(
                'ingest failed job=%d %s after %.1fs: %s: %s',
                job.id,
                job.label,
                time.monotonic() - t0,
                type(e).__name__,
                e,
            )
        finally:
            self.in_progress -= 1
        if job.on_failure is not None:
            try:
                await asyncio.wait_for(job.on_failure(self.graphiti), timeout=60)
            except Exception as e:
                logger.error(
                    'ingest cleanup failed job=%d %s: %s: %s',
                    job.id,
                    job.label,
                    type(e).__name__,
                    e,
                )

    async def worker(self, worker_id: int):
        while True:
            job = await self.queue.get()  # CancelledError here ends the worker
            try:
                await self._run_one(worker_id, job)
            except asyncio.CancelledError:
                raise
            except BaseException:  # never let anything kill the worker loop
                logger.exception('ingest worker %d: unexpected error', worker_id)
            finally:
                self.queue.task_done()

    async def start(self):
        settings = get_settings()
        self.timeout = settings.ingest_job_timeout
        self.graphiti = create_graphiti(settings)
        n = settings.ingest_concurrency
        self.tasks = [asyncio.create_task(self.worker(i + 1)) for i in range(n)]
        logger.info('ingest worker started: concurrency=%d job_timeout=%.0fs', n, self.timeout)

    async def stop(self):
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        dropped = self.queue.qsize()
        while not self.queue.empty():
            self.queue.get_nowait()
        if dropped:
            logger.warning('ingest worker stopped: %d queued job(s) dropped', dropped)
        if self.graphiti is not None:
            await self.graphiti.close()

    def status(self) -> dict:
        return {
            'queued': self.queue.qsize(),
            'in_progress': self.in_progress,
            'workers': len(self.tasks),
            'job_timeout_s': self.timeout,
            **self.stats,
        }


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await async_worker.start()
    yield
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


@router.get('/ingest/status', status_code=status.HTTP_200_OK)
async def ingest_status():
    return async_worker.status()


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(request: AddMessagesRequest):
    for m in request.messages:
        await async_worker.put(_message_job(request.group_id, m))
    return Result(message='Messages added to processing queue', success=True)


def _message_job(group_id: str, m: Message) -> Job:
    episode_body = f'{m.role or ""}({m.role_type}): {m.content}'
    created = {'episode': False}

    async def run(graphiti: ZepGraphiti):
        if m.uuid is not None:
            created['episode'] = await graphiti.ensure_episode(
                m.uuid, group_id, m.name, episode_body, m.timestamp, m.source_description
            )
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=group_id,
            name=m.name,
            episode_body=episode_body,
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )

    async def on_failure(graphiti: ZepGraphiti):
        # Don't leave a half-processed placeholder behind: the client should see the episode
        # as missing and re-send it.
        if m.uuid is not None and created['episode']:
            await graphiti.remove_episode(m.uuid)
            logger.info('ingest removed unprocessed episode %s', m.uuid)

    label = f'group={group_id} uuid={m.uuid} name={m.name!r} chars={len(m.content)}'
    return Job(id=async_worker.next_id(), label=label, run=run, on_failure=on_failure)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
