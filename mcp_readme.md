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

## Open Questions
- Which core functions should be exposed first?
- Do we need auth/ACLs at MCP boundary?
- Should MCP tools be versioned (e.g., `v1` namespace)?
- How should we handle streaming responses (if required)?

## Next Steps
1. Confirm the list of core functions to expose.
2. Define MCP tool schemas and response shapes.
3. Implement thin wrappers in `RAG/mcp/tools.py`.
4. Add tests (unit + integration).

## Implementation Plan (Initial Slice: Knowledge Base)
### Phase 0: Package scaffolding
- Add `RAG/mcp/` subpackage with `app.py`, `tools.py`, and `schemas.py`.
- Add `RAG/services/` subpackage with `kb_service.py` for Streamlit-free logic.

### Phase 1: Stateless KB tools (first vertical slice)
- Implement service functions:
  - `build_kb(...)` writes FAISS index + metadata to disk and optionally runs GraphRAG.
  - `load_kb(...)` validates and inspects an existing KB on disk.
  - `append_kb(...)` appends new PDFs to an existing KB and updates GraphRAG.
- MCP tools wrap the service functions with minimal input validation.
- Outputs include summary fields and warnings (no Streamlit session state).

### Phase 2: Retrieval + answer tools (next slice)
- Implement `retrieve.context` to return both `context_text` and `sources`.
- Implement `generate_answer` using the existing streaming/generation core.
- Implement a combined tool `retrieve_and_answer` that calls the two in sequence.

### Phase 3: Tests
- Unit tests for service functions (happy path + error path).
- MCP wrapper tests for schema validation and error translation.
- One integration test that runs the MCP app and hits `kb.build`.

## Status Update (What’s Done)
- Created MCP scaffolding: `RAG/mcp/app.py`, `RAG/mcp/tools.py`, `RAG/mcp/schemas.py`.
- Added Streamlit-free KB service layer in `RAG/services/kb_service.py` with build/load/append.
- Added Streamlit-free retrieval and generation services in `RAG/services/retrieval_service.py` and `RAG/services/generation_service.py`.
- Added smoke scripts:
  - `scripts/mcp_smoke_kb.py` (local `load_kb` service)
  - `scripts/mcp_smoke_kb_build.py` (build KB, requires API key)
  - `scripts/mcp_smoke_kb_append.py` (append KB, requires API key)
  - `scripts/mcp_smoke_http_kb_load.py` (HTTP MCP call to `kb_load`)
  - `scripts/mcp_smoke_retrieve_context.py` (HTTP MCP call to `retrieve_context_tool`)
  - `scripts/mcp_smoke_retrieve_and_answer.py` (HTTP MCP call to `retrieve_and_answer_tool`)
- Added FastMCP dependency in `requirements.txt`.

## Smoke Scripts (Commands)
### Environment setup
```
source .venv/bin/activate
export OPENAI_API_KEY=...
```

### Local service smoke (no MCP)
```
python3 scripts/mcp_smoke_kb.py
```

### KB build (requires OpenAI API key)
```
python3 scripts/mcp_smoke_kb_build.py
```

### KB append (requires OpenAI API key)
```
python3 scripts/mcp_smoke_kb_append.py
```

### MCP HTTP: kb_load
```
python3 scripts/mcp_smoke_http_kb_load.py
```

### MCP HTTP: retrieve_context_tool
```
export KB_INDEX_PATH=/path/to/index.faiss
export KB_META_PATH=/path/to/meta.pkl
export KB_GRAPHRAG_DIR=/path/to/graphrag  # optional
python3 scripts/mcp_smoke_retrieve_context.py
```

### MCP HTTP: retrieve_and_answer_tool
```
export KB_INDEX_PATH=/path/to/index.faiss
export KB_META_PATH=/path/to/meta.pkl
export KB_GRAPHRAG_DIR=/path/to/graphrag  # optional
export QA_MODEL=gpt-4o-mini
python3 scripts/mcp_smoke_retrieve_and_answer.py
```

### MCP HTTP: generate_answer_tool
```
export QA_MODEL=gpt-4o-mini
python3 scripts/mcp_smoke_generate_answer.py
```


# MCP Insepctor:
1. Start the MCP server (HTTP):

python3 -c "from RAG.mcp.app import create_app; create_app().run(transport='http', host='127.0.0.1', port=8000)"

2. In MCP Inspector web UI, add a server:
npx @modelcontextprotocol/inspector

http://127.0.0.1:8000/mcp/

{
  "query": "What does this KB cover?",
  "index_path": "/data/Work/ORNL_internship/test_folder/test_index.index",
  "meta_path": "/data/Work/ORNL_internship/test_folder/test_metadata.pkl",
  "graphrag_dir": "/data/Work/ORNL_internship/test_folder/graph_kb",
  "api_key": null,
  "diversity": 0.3,
  "top_k_faiss": 5,
  "use_graphrag": true,
  "retriever_model": "gpt-4o-mini",
  "compressor_model": "gpt-4o-mini",
  "model": "gpt-4o-mini",
  "system": "You are a helpful assistant.",
  "enable_web_search": false
}


## TODO
1. integrate mcp with gpt
	1. The user would be required to use OpenAI Platforms
	2. OpenAI Platforms already has RAG implements
		1. Still requires user to make/insert keys
	3. ChatGPT already has ACEKnowledgeGraphs

- MCP vs Web App (I think the latter is preferred, and we just make a more catered towards the materials science use case)
	1. Setup cloud compute would be very cool for me
		- Dockerize Streamlit app
		- Push image to ECR
		- ECS Fargate service (1 task to start)
		- ALB + ACM cert + Route 53
		- Store secrets in Secrets Manager
		- Log to CloudWatch
		- Put user uploads in **S3** (instead of local disk)
	2. Host streamlit app on server:
		1. Reverse proxy (Nginx or Caddy)
		2. Authentication: SSO or App-level login
		3. Implement some sort of persistent session store
		4. Containerize multiple streamlit workers (can create some kind of ELB if needed)

2. knowledge base benchmarking
3. create pytest suite
4. Come up with something for Stephen

5. yongtaoliu.github.io/aecroscopy.pyae



