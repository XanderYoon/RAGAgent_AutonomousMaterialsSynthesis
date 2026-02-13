
---
# Refactoring
## Purpose
Break up the original monolithic `app.py` (1000+ lines) into focused modules so core logic is reusable across UI, CLI, and MCP while improving testability and maintainability.

## Scope
- Extract KB lifecycle, retrieval, and generation logic into service modules.
- Isolate UI wiring from core logic.
- Ensure new modules are importable by MCP tools without Streamlit state.

## Rationale: Modular vs. Monolith
### Recommended: modular services + thin interfaces
- **Separation of concerns:** core logic lives in `RAG/services/`, UI and MCP are thin wrappers.
- **Testability:** services can be unit-tested without UI/session state.
- **Reuse:** the same core is shared by Streamlit, MCP, and scripts.
- **Clarity:** each module has a single responsibility.

### Alternative: keep everything in `app.py`
- Faster to prototype but difficult to test, reuse, and maintain.

Given the MCP integration and growing feature set, modularization is required.

## Proposed/Current Layout
```
RAG/
  services/
    kb_service.py
    retrieval_service.py
    generation_service.py
  mcp/
    app.py
    tools.py
    schemas.py
  ui/                     # Streamlit UI wiring
    ...
```

## Design Principles
- **Thin interfaces:** UI/MCP call service functions without embedding logic.
- **No session coupling:** services avoid Streamlit session state.
- **Explicit inputs/outputs:** service functions return structured data.
- **Testable by default:** services are written for unit tests first.

## Pydantic Notes
- Use Pydantic models at module boundaries to validate inputs and standardize outputs.
- Keep validation in schemas (e.g., MCP layer) while services operate on typed data.

## Refactor Summary
- Moved KB build/load/append logic into `RAG/services/kb_service.py`.
- Moved retrieval logic into `RAG/services/retrieval_service.py`.
- Moved generation logic into `RAG/services/generation_service.py`.
- Reduced `app.py` to UI orchestration and wiring to services.

---
# MCP Overview and Design

## Purpose
The Model Context Protocol (MCP) layer will expose a stable, external-facing interface for this project via FastMCP. It should provide a clean integration surface while keeping core logic unchanged and reusable by other interfaces (UI, CLI, batch jobs). This is intended to make the system more accessible to other researchers by offering a consistent, documented API surface.

## Scope
- Provide MCP tools that map to existing core capabilities.
- Keep MCP wiring, schema validation, and error translation isolated from core logic.
- Avoid duplicating business logic: MCP calls core functions and adapts inputs/outputs.

## Rationale: Mirror vs. Embed
### Recommended: Mirror core functions in an MCP subpackage
- **Separation of concerns:** MCP-specific concerns (schemas, auth, observability) stay out of core.
- **Stable interface:** MCP tools can be versioned independently from core changes.
- **Safer boundary:** easier to validate inputs and constrain behavior for external callers.
- **Researcher accessibility:** a dedicated MCP layer provides a clear, documented entry point for other researchers without requiring them to navigate internal modules or UI-specific code.

### Alternative: Embed MCP directly in core
- Lower boilerplate for small projects.
- MCP concerns bleed into core and can complicate maintenance.

Given MCP is an integration boundary, a thin “mirror” layer is preferred.

## Proposed Package Layout
```
RAG/
  core/                  # existing core logic (or current module locations)
  generation/
  ingestion/
  ...
  mcp/
    __init__.py
    app.py               # FastMCP app initialization
    tools.py             # MCP tool definitions (thin wrappers)
    schemas.py           # Pydantic/typed schemas if needed
    errors.py            # Error translation and normalization
    logging.py           # MCP-specific logging/telemetry
```

## Design Principles
- **Thin wrappers:** MCP tools should call core functions without altering logic.
- **Explicit schemas:** Validate inputs and define clear outputs.
- **Graceful errors:** Convert core exceptions into MCP-friendly errors.
- **No side effects:** MCP layer should not mutate core behavior.
- **Testable:** Unit tests for wrappers, integration tests for MCP server.

## Tool Mapping Strategy
- Each MCP tool maps 1:1 to a core function or small core workflow.
- If a tool needs orchestration, create a small core “service” function and call that.
- Avoid business logic inside MCP.

## Error Handling
- Map core exceptions to MCP error codes/messages.
- Include minimal diagnostic context without leaking sensitive details.
- Normalize error shapes for client consistency.

## Configuration
- MCP config should live alongside other configuration (e.g., env vars or `config.py`).
- Keep MCP-specific settings (auth, rate limits, allowed models) isolated in MCP layer.

## Testing Plan
### Unit tests
- Validate schema parsing and input coercion.
- Ensure wrappers pass the correct args to core functions.
- Verify error mapping.

### Integration tests
- Spin up FastMCP app and call tools end-to-end.
- Smoke tests for common paths and expected outputs.
- Negative tests for invalid input and unauthorized access (if applicable).

## MCP Inspector
- Run a local npx MCP Inspector session to interact with the MCP API and see tool usage/performance. 

---
# Testing Suite 
## Purpose
Ensure core services and MCP tooling behave correctly and remain stable as the system evolves.

## Scope
- Unit tests for service functions (happy path + error path).
- MCP wrapper tests for schema validation and error translation.
- Integration test that runs the MCP app and hits `kb.build`.

## Rationale: Pytest suite vs. ad-hoc scripts
### Recommended: Pytest-first with targeted smoke scripts
- **Repeatable:** consistent runs across environments via `pytest.ini`.
- **Focused:** unit tests isolate service logic without UI state.
- **Coverage:** MCP tools are validated end-to-end in integration tests.

### Alternative: rely on manual smoke scripts only
- Faster for quick checks but not comprehensive or automated.

To run pytests: `pytest -q tests/`

## Current Layout
```
tests/                 # unit + integration tests
pytest.ini             # shared pytest config
```

## Design Principles
- **Isolated:** tests avoid Streamlit session state.
- **Deterministic:** minimize external dependencies and flakiness.
- **Representative:** cover core service paths and MCP wrappers.

---
# TODO

### Need to review:
- https://ieeexplore.ieee.org/document/10600957
- https://arxiv.org/abs/2412.17364
- https://arxiv.org/abs/2410.12890