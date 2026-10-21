# plosson/graphiti — `houlahop` branch

Fork of [getzep/graphiti](https://github.com/getzep/graphiti) based on tag `v0.31.2`
(commit d72407d, the same server code as the `zepai/graphiti:0.31.2` image).
Only the REST server (`server/`) and the root `Dockerfile` are changed; `graphiti_core`
is untouched and the image still installs `graphiti-core` from PyPI.

## Changes
1. **FalkorDB auth** — the server passes `FALKORDB_USERNAME` / `FALKORDB_PASSWORD` (plus the
   existing `FALKORDB_HOST` / `FALKORDB_PORT` / `FALKORDB_DATABASE`) to `FalkorDriver`.
2. **OpenAI-compatible providers (OpenRouter, vLLM, …)**
   - `LLM_BASE_URL` (falls back to `OPENAI_BASE_URL`): LLM = `OpenAIGenericClient`
     (`chat.completions` + `json_schema`) with `MODEL_NAME` / `SMALL_MODEL_NAME`;
     reranker = `OpenAIRerankerClient` on the same endpoint with `RERANKER_MODEL_NAME`
     (needs a model with `logprobs` + `logit_bias`, e.g. `openai/gpt-4o-mini` on OpenRouter).
   - `EMBEDDER_BASE_URL` (falls back to `OPENAI_BASE_URL`): `OpenAIEmbedder` with
     `EMBEDDING_MODEL_NAME`. Set `EMBEDDING_DIM` to the model's size (1536 for
     `text-embedding-3-small`; graphiti-core defaults to 1024 and truncates).
   - `LLM_STRUCTURED_OUTPUT_MODE`: `json_schema` (default) or `json_object`. Use `json_object`
     on OpenRouter: its OpenAI route enforces strict schemas, which graphiti's models don't meet.
   - `OPENAI_API_KEY` is the provider key. With no base URL set, behaviour is unchanged (OpenAI).
3. **Ingest worker** — a failing `/messages` background job is logged and the queue keeps going
   (upstream, one exception silently killed the worker and all later episodes stayed queued).
4. **Client-chosen episode uuid** — `/messages` items with a `uuid` create a new episode with
   that uuid (graphiti_core's `add_episode(uuid=...)` only re-processes existing episodes and
   failed with `NodeNotFoundError`).
5. **One FalkorDB graph** — by default (`FALKORDB_SINGLE_GRAPH=true`) every group_id lives in the
   `FALKORDB_DATABASE` graph, filtered by the `group_id` property. Upstream puts each group in
   its own graph, which the server's by-uuid endpoints (`GET/DELETE /entity-edge`,
   `DELETE /episode`, `DELETE /group`) never look at.
7. **Bounded provider calls** — with an OpenAI-compatible base URL, the LLM, reranker and embedder
   share bounded `AsyncOpenAI` clients: `LLM_TIMEOUT` (default 60 s per HTTP request, connect 10 s)
   and `LLM_MAX_RETRIES` (default 1). `LLM_MAX_TOKENS` (default 8192) caps completion tokens
   (OpenAIGenericClient defaults to 16384). The SDK default (600 s × 3 attempts) let one stalled
   request hold the ingest worker for up to ~30 min.
8. **Ingest worker hardening** (`server/graph_service/routers/ingest.py`)
   - Jobs run on a long-lived Graphiti client owned by the worker; upstream reused the
     request-scoped client, which is closed as soon as the 202 response is sent.
   - Each job is wrapped in `asyncio.wait_for(..., INGEST_JOB_TIMEOUT)` (default 180 s); a
     timeout or any exception is logged and the worker moves on to the next job.
   - On failure/timeout the placeholder episode created for a client uuid is removed
     (`remove_episode`), so the client sees it as missing and can re-send it.
   - `INGEST_CONCURRENCY` (default 1, max 8) worker tasks.
   - One log line per job: `ingest queued|start|done|failed|timeout job=N group=… uuid=… name=… chars=…`
     with wait time, duration and queue size; `LOG_LEVEL` (default INFO) via `logging.basicConfig`.
   - `GET /ingest/status` → `{queued, in_progress, workers, job_timeout_s, done, failed, timeout}`.
6. **Dockerfile** — `INSTALL_FALKORDB` defaults to `true`, so the image includes the FalkorDB driver.

## Build
    docker build --build-arg GRAPHITI_VERSION=0.31.2 -t graphiti-server .

## OpenRouter example
    OPENAI_API_KEY=sk-or-...  OPENAI_BASE_URL=https://openrouter.ai/api/v1
    MODEL_NAME=openai/gpt-4.1-mini  EMBEDDING_MODEL_NAME=openai/text-embedding-3-small
    RERANKER_MODEL_NAME=openai/gpt-4o-mini  EMBEDDING_DIM=1536  LLM_STRUCTURED_OUTPUT_MODE=json_object
