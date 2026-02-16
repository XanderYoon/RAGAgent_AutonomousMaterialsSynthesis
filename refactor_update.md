# Refactor Progress Log

## Change 1: Added new core RAG modules
- Created `RAG/kb_builder.py`.
  - Consolidated KB build/load/append logic into function-based core APIs.
  - Added domain exceptions in this module: `KnowledgeBaseError`, `KnowledgeBaseBuildError`, `KnowledgeBaseLoadError`, `KnowledgeBaseAppendError`, `GraphRAGWorkspaceError`.
  - Kept stable return dictionaries for build/load/append flows.
- Created `RAG/retrieval.py`.
  - Added core `retrieve_context(...)` function.
  - Added `RetrievalError` and explicit index/metadata load error wrapping.
- Created `RAG/generation.py`.
  - Added core `generate_answer(...)` function.
  - Added `GenerationError` and wrapped model call failures.

Status:
- New core files exist and are ready for caller rewiring.
- Old service and adapter modules are not yet rewired in this step.

## Change 2: Converted service layer to compatibility wrappers
- Replaced implementation-heavy files with thin re-export wrappers:
  - `RAG/services/kb_service.py` now re-exports APIs/exceptions from `RAG/kb_builder.py`.
  - `RAG/services/retrieval_service.py` now re-exports from `RAG/retrieval.py`.
  - `RAG/services/generation_service.py` now re-exports from `RAG/generation.py`.

Why:
- Removes duplicated core pipeline logic while keeping backward-compatible import paths during migration.

Status:
- Existing callers importing from `RAG.services.*` continue to work.
- Core implementation source of truth is now `RAG/*.py`.

## Change 3: Rewired UI adapters to call core RAG modules
- Refactored `ui/kb_setup.py`:
  - Removed dependency on `KnowledgeBaseBuilder` class in `RAG/retrieval/kb_builder.py`.
  - Now calls core functions from `RAG/kb_builder.py`: `build_kb`, `load_kb`, `append_kb`.
  - Added `_register_kb_in_session(...)` in UI layer to load FAISS+metadata and update Streamlit session state.
  - Session registration now also persists `index_path` and `meta_path` for downstream retrieval calls.
- Refactored `ui/qa_panel.py`:
  - Removed dependency on `QAContextRetriever` class.
  - Now calls `RAG/retrieval.py::retrieve_context(...)` for KB retrieval.
  - Now calls `RAG/generation.py::generate_answer(...)` for final answer generation.
  - Kept upload processing in UI layer and appends upload-derived context in adapter code.
- Updated `state/session.py` defaults:
  - Added `index_path` and `meta_path` keys.

Status:
- UI now uses the new core RAG modules for KB/retrieval/generation logic.

## Change 4: Moved MCP package to root and rewired imports
- Moved directory:
  - `RAG/mcp/` -> `MCP/`
- Updated MCP package internals:
  - `MCP/__init__.py` imports from `MCP.app`.
  - `MCP/app.py` imports from `MCP.tools`.
  - `MCP/tools.py` now imports schemas from `MCP.schemas`.
- Rewired MCP tools to call new core modules directly:
  - `RAG.kb_builder`
  - `RAG.retrieval`
  - `RAG.generation`
- Updated MCP startup and tests:
  - `run_mcp_demo_unix.sh` now boots `MCP.app.create_app`.
  - `tests/conftest.py` now imports `create_app` from `MCP.app`.

Status:
- MCP is now a root-level adapter package and no longer nested under `RAG/`.

## Change 5: Flattened `RAG/` layout and removed module/package collisions
- Updated tests to target new core KB module directly:
  - `tests/test_kb_service.py` now imports from `RAG.kb_builder`.
  - `tests/conftest.py` now imports `build_kb` from `RAG.kb_builder`.
- Resolved `RAG/retrieval.py` conflict with legacy `RAG/retrieval/` package:
  - Copied GraphRAG helper to `RAG/graphrag_query.py`.
  - Updated `RAG/retrieval.py` to import from `RAG.graphrag_query`.
  - Removed legacy directory `RAG/retrieval/`.
- Resolved `RAG/generation.py` conflict with legacy `RAG/generation/` package:
  - Removed legacy directory `RAG/generation/` (previous `streaming.py` path no longer used).
- Minor UI alignment:
  - `ui/kb_setup.py` now uses `TOP_K_TEXT_BM25` constant when building BM25 retriever in session.

Status:
- `RAG/` now has flat core modules as designed: `kb_builder.py`, `retrieval.py`, `generation.py`.

## Change 6: Added core-module focused tests
- Added `tests/test_generation.py`:
  - Validates `RAG.generation.generate_answer(...)` fails fast with `GenerationError` when no API key is provided.
- Added `tests/test_retrieval.py`:
  - Validates `RAG.retrieval.retrieve_context(...)` fails fast with `RetrievalError` when no API key is provided.

Status:
- Basic non-network failure-path coverage now exists for the new core generation/retrieval modules.

## Change 7: Removed deprecated service package
- Deleted `RAG/services/` after confirming no remaining code imports reference it.

Status:
- Core pipeline logic source of truth is now exclusively in:
  - `RAG/kb_builder.py`
  - `RAG/retrieval.py`
  - `RAG/generation.py`

## Change 8: Updated docs and test layout
- Updated `README.md` repository structure block to reflect flat core modules in `RAG/` and root `MCP/`.
- Updated `AGENTS.md` architecture section to match the new design:
  - Core logic centralized in `RAG/*.py`.
  - MCP as root-level adapter package.
- Renamed `tests/test_kb_service.py` -> `tests/test_kb_builder.py` to match new core module naming.

Validation:
- Ran compile checks with `python3 -m py_compile` across changed modules; all passed.
- Attempted to run test suite with `python3 -m pytest -q`, but pytest is not installed in this environment (`No module named pytest`).

## Change 9: Additional validation attempt
- Tried running tests via project venv interpreter (`.venv/bin/python -m pytest -q`).
- Result: pytest is also missing in the venv (`No module named pytest`).

Status:
- Validation in this environment remains limited to compile checks.
