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
   - `OPENAI_API_KEY` is the provider key. With no base URL set, behaviour is unchanged (OpenAI).
3. **Dockerfile** — `INSTALL_FALKORDB` defaults to `true`, so the image includes the FalkorDB driver.

## Build
    docker build --build-arg GRAPHITI_VERSION=0.31.2 -t graphiti-server .

## OpenRouter example
    OPENAI_API_KEY=sk-or-...  OPENAI_BASE_URL=https://openrouter.ai/api/v1
    MODEL_NAME=openai/gpt-4.1-mini  EMBEDDING_MODEL_NAME=openai/text-embedding-3-small
    RERANKER_MODEL_NAME=openai/gpt-4o-mini  EMBEDDING_DIM=1536
