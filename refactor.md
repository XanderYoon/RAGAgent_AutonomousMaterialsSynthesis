# RAG Refactor Design and Implementation Plan

## 1. Objective
Unify duplicated RAG logic so both UI and MCP call the same core pipeline functions.

Target outcome:
- Keep core logic only in `RAG/generation.py`, `RAG/retrieval.py`, and `RAG/kb_builder.py`.
- Move MCP package out of `RAG/` to root-level `MCP/`.
- Keep UI and MCP as thin interface adapters that only validate input, map callbacks, and call core RAG functions.
- Ensure future pipeline changes are made once in `RAG/` and propagate to both UI and MCP.

## 2. Current State and Problems
Current duplication and coupling:
- UI path uses class-based logic in `RAG/retrieval/kb_builder.py` and `RAG/retrieval/retriever.py`.
- MCP path uses function-based logic in `RAG/services/kb_service.py`, `RAG/services/retrieval_service.py`, `RAG/services/generation_service.py`.
- MCP implementation is nested under `RAG/mcp/`, mixed with domain/core package.

Issues this creates:
- Behavioral drift risk between UI and MCP.
- Two places to edit for one pipeline change.
- Extra maintenance/testing overhead.

## 3. Design Principles (AGENTS.md alignment)
- Single responsibility: core functions do RAG work only; adapters do transport/UI work only.
- Open-closed: extend through new helper functions and optional keyword args with safe defaults.
- Interface segregation: pass only required dependencies (`client`, `embeddings`, `callbacks`, config).
- Dependency inversion: core modules avoid Streamlit/MCP framework dependencies.
- Stable contracts: explicit return dictionaries with fixed keys at boundaries.

## 4. Target Architecture

```text
.
├── RAG/
│   ├── kb_builder.py        # build/load/append knowledge base
│   ├── retrieval.py         # hybrid retrieval + optional GraphRAG query
│   ├── generation.py        # answer generation
│   └── ingestion/           # loaders/cleaners/chunkers/embeddings/faiss/graphrag helpers
├── ui/                      # Streamlit adapters
│   ├── kb_setup.py
│   └── qa_panel.py
├── MCP/                     # MCP adapters and schemas (moved from RAG/mcp)
│   ├── app.py
│   ├── tools.py
│   ├── schemas.py
│   └── __init__.py
└── tests/
    ├── test_kb_builder.py
    ├── test_retrieval.py
    ├── test_generation.py
    └── test_mcp_tools_http.py
```

## 5. Core Module Contracts

### `RAG/kb_builder.py`
Public API:
- `build_kb(*, client, embeddings, enc, pdf_dir, index_path, meta_path, graphrag_dir, embedding_model, run_graphrag=True, api_key=None, callbacks=None) -> dict`
- `load_kb(*, index_path, meta_path, graphrag_dir) -> dict`
- `append_kb(*, client, enc, index_path, meta_path, append_folder, graphrag_dir, run_graphrag=True, api_key=None, callbacks=None) -> dict`

Stable return keys:
- build: `total_chunks`, `total_tokens`, `estimated_cost`, `warnings`
- load: `status`, `embedding_model`, `dimension`, `chunk_count`, `graphrag_dir`
- append: `new_chunks`, `new_tokens`, `estimated_cost`, `warnings`

Domain errors:
- `KnowledgeBaseError`
- `KnowledgeBaseBuildError`
- `KnowledgeBaseLoadError`
- `KnowledgeBaseAppendError`
- `GraphRAGWorkspaceError`

Notes:
- Migrate and consolidate logic from `RAG/services/kb_service.py` and relevant reusable parts of `RAG/retrieval/kb_builder.py`.
- Remove Streamlit session-state mutation from core logic.

### `RAG/retrieval.py`
Public API:
- `retrieve_context(*, query, index_path, meta_path, graphrag_dir, api_key=None, diversity=0.3, top_k_faiss=None, use_graphrag=True, retriever_model="gpt-4o-mini", compressor_model="gpt-4o-mini") -> dict`

Stable return keys:
- `context_text`, `sources`, `embedding_model`

Domain errors:
- `RetrievalError` (new)
- wrap lower-level failures with actionable messages and `raise ... from exc`

Notes:
- Start from `RAG/services/retrieval_service.py` as the base.
- Keep optional GraphRAG inclusion behavior.

### `RAG/generation.py`
Public API:
- `generate_answer(*, query, context_text, model, system, api_key=None, enable_web_search=False) -> str`

Domain errors:
- `GenerationError` (new)

Notes:
- Migrate from `RAG/services/generation_service.py`.
- Keep prompt construction private and deterministic.

## 6. Adapter Responsibilities

### UI (`ui/`)
- `ui/kb_setup.py`
  - Validate user inputs (`KBBuildRequest`, `KBLoadRequest`, `KBAppendRequest`).
  - Call `RAG.kb_builder` functions.
  - Update `st.session_state` with loaded artifacts/results.
  - Map callbacks to Streamlit display methods.

- `ui/qa_panel.py`
  - Use `RAG.retrieval.retrieve_context` and `RAG.generation.generate_answer` (or current streaming wrapper if retained).
  - Keep upload handling in adapter layer; do not move Streamlit-only logic into core.

### MCP (`MCP/`)
- `MCP/tools.py`
  - Validate MCP payloads via `MCP/schemas.py`.
  - Call the same `RAG` core functions.
  - Convert core outputs to schema responses.

- `MCP/app.py`
  - FastMCP app composition only.

## 7. File Move and Deprecation Map
- Move `RAG/mcp/*` -> `MCP/*`.
- Replace imports:
  - `from RAG.mcp...` -> `from MCP...`
  - `from RAG.services.kb_service...` -> `from RAG.kb_builder...`
  - `from RAG.services.retrieval_service...` -> `from RAG.retrieval...`
  - `from RAG.services.generation_service...` -> `from RAG.generation...`
- Deprecate/remove after migration:
  - `RAG/services/`
  - `RAG/retrieval/retriever.py` (if functionality fully moved to `RAG/retrieval.py`)
  - `RAG/retrieval/kb_builder.py` (class-based UI path)

## 8. Phased Implementation Plan

### Phase 0: Safety Baseline
1. Snapshot current tests and key smoke paths.
2. Record current public function signatures and return payloads.

### Phase 1: Introduce New Core Modules
1. Create `RAG/kb_builder.py` from service implementation + extracted reusable helpers.
2. Create `RAG/retrieval.py` from service implementation.
3. Create `RAG/generation.py` from service implementation.
4. Add/adjust concise docstrings and boundary validation.

### Phase 2: Rewire UI to Core
1. Update `ui/kb_setup.py` imports and calls to `RAG.kb_builder`.
2. Update `ui/qa_panel.py` retrieval/generation calls to new core modules.
3. Keep UI-only concerns in `ui/` (session state, progress UI, uploads).

### Phase 3: Move and Rewire MCP
1. Move `RAG/mcp` to root `MCP`.
2. Update `MCP/tools.py` to call `RAG` core modules.
3. Update startup paths:
   - `run_mcp_demo_unix.sh`
   - test fixtures in `tests/conftest.py`

### Phase 4: Remove Deprecated Layers
1. Remove or freeze `RAG/services/` and old retrieval class-based files once green.
2. Remove stale imports and dead code.

### Phase 5: Validation and Cleanup
1. Run unit tests and MCP HTTP tests.
2. Verify Streamlit build/load/append/retrieve/generate flows manually.
3. Update `README.md` and architecture notes.

## 9. Testing Strategy

Core tests:
- `tests/test_kb_builder.py`
  - build success/failure (no PDFs, invalid model, empty chunks)
  - load success/failure (missing files, empty metadata)
  - append success/failure (missing folder/files, no new chunks)
- `tests/test_retrieval.py`
  - retrieval result shape
  - API key required
  - graphrag optional behavior and error handling
- `tests/test_generation.py`
  - prompt/response path
  - API key required

Adapter tests:
- `tests/test_mcp_tools_http.py`
  - update imports and ensure tool names/response schemas remain stable.

Regression checks:
- build/load/append parity between UI and MCP using same core module outputs.

## 10. Risks and Mitigations
- Risk: Import breakage during file moves.
  - Mitigation: phase in new modules first, then switch callers, then delete old modules.
- Risk: behavior mismatch between previous UI class path and function path.
  - Mitigation: normalize on one core implementation and compare outputs in tests.
- Risk: session-state coupling leaks into core.
  - Mitigation: enforce no Streamlit imports in `RAG/*.py` core modules.

## 11. Open Decisions
1. Directory casing:
   - Current project uses `ui/` lowercase.
   - Requested target mentions `/UI/` and `/MCP/`.
   - Recommendation: keep `ui/` lowercase for minimal disruption; create `MCP/` as requested.
2. Streaming generation in UI:
   - Keep `RAG/generation/streaming.py` as a UI helper, or unify into `RAG/generation.py` with optional streaming mode.

## 12. Definition of Done
- UI and MCP both call only `RAG/kb_builder.py`, `RAG/retrieval.py`, and `RAG/generation.py` for core pipeline behavior.
- `RAG/mcp/` no longer exists; MCP runs from root `MCP/`.
- No duplicated KB/retrieval/generation logic across UI/MCP.
- Tests pass for core modules and MCP HTTP tool flows.
- Documentation updated to reflect final architecture.
