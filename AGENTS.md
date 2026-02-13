# AGENT GUIDE

## App design (current architecture)
- `app.py` is the Streamlit composition root.
- `ui/` contains presentation and user interaction only (`API_setup.py`, `kb_setup.py`, `qa_panel.py`).
- `state/` centralizes config constants, session bootstrap, and shared schemas.
- `RAG/services/` orchestrates end-to-end use cases:
  - `kb_service.py`: build/load/append knowledge base and optional GraphRAG indexing.
  - `retrieval_service.py`: hybrid retrieval pipeline (FAISS + BM25 + multi-query + compression + optional GraphRAG query).
  - `generation_service.py`: final LLM answer generation.
- `ingestion/` provides data-processing primitives (load, clean, chunk, embed, vector store utilities).
- `RAG/retrieval/` and `RAG/generation/` contain lower-level retrieval/generation helpers used by services.
- `RAG/mcp/` exposes MCP-compatible tools and schemas.
- `tests/` validates behavior at service and MCP-tool boundaries.

Design intent: keep UI thin, keep business logic in services, keep data/ML operations in ingestion/retrieval modules, and keep shared contracts in `state/`.

## Function style guide (SOLID)

### Single responsibility
- Each function should do one cohesive task.
- Prefer extraction over long functions; move parsing/validation/IO/transform/model-call into separate helpers.

### Open-closed
- Extend behavior via new functions/classes and dependency injection, not by editing stable code paths.
- Add optional arguments with safe defaults when introducing non-breaking behavior.

### Liskov substitution
- Keep return types and side effects predictable across interchangeable implementations.
- Substitutable components (e.g., retrievers, model providers) must preserve expected contracts.

### Interface segregation
- Pass only required dependencies (`client`, `embeddings`, `callbacks`) instead of large state bags.
- Keep callbacks focused (`info`, `progress`) and optional.

### Dependency inversion
- High-level orchestration depends on abstractions/contracts, not low-level implementation details.
- Inject external clients and configuration; avoid hidden global state where possible.

## Naming and signatures
- Use `snake_case` for functions, variables, parameters, and modules.
- Keep public service function signatures keyword-only (`*`) when argument order could be ambiguous.
- Prefer explicit return dictionaries with stable keys for UI/service boundaries.
- Ruff linter formatting

## Docstring standard (concise and explicit)
Use short docstrings with clear inputs, output, and behavior.

For complex functions or a public interface:
```python
def function_name(arg_one: str, arg_two: int) -> dict:
    """Do one specific task.

    Args:
        arg_one: Meaning and expected format.
        arg_two: Meaning and constraints.

    Returns:
        Dictionary with stable keys and value types.

    Raises:
        DomainError: When validation or downstream execution fails.
    """
```

For simpler functions, one line descriptors are sufficient
```python
def function_name(arg_one: str, arg_two: int) -> dict:
    """Validate and inspect an existing knowledge base on disk."""
```

Rules:
- Keep to 5-12 lines unless complexity requires more.
- Describe constraints (required, allowed range, file existence, model support).
- Document non-obvious side effects (file writes, network calls, state mutation).

## Error handling rules
- Validate inputs at function boundaries and fail fast.
- Raise specific, domain-oriented exceptions (`KnowledgeBaseBuildError`, `KnowledgeBaseLoadError`, etc.) for service-layer failures.
- Wrap low-level exceptions with context using `raise ... from exc`.
- Error messages must be actionable and include what failed, where, and expected state.

Preferred message style:
- `"No PDFs found in directory: {pdf_dir}"`
- `"Unsupported embedding model '{embedding_model}'. Supported: {sorted_models}"`
- `"Failed to load metadata file at {meta_path}."`

Avoid:
- Silent `except` blocks unless explicitly intentional and documented.
- Generic messages like `"Something went wrong"`.

## Practical implementation checklist
- Add/adjust docstrings for new or changed public functions.
- Keep functions small; extract helpers when branching/IO grows.
- Preserve `snake_case` naming.
- Add explicit validation and informative exception messages.
- Add/extend tests for success and failure paths.

## Task Execution Protocol

When assigned a task, the agent must:

1. Restate the objective.
2. Identify relevant modules.
3. Propose implementation plan.
4. Implement code.
5. Write tests.
6. Explain decisions.
7. Confirm compliance with guide.
