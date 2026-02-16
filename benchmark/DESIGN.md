# Benchmarking Suite Design (Service-Reuse + Pluggable Judges)

## 1. Objective
Design a script-first benchmark suite under `benchmark/` that reuses the same service-layer functions used by UI/MCP for:
- knowledge base build/load,
- retrieval + answer generation,
- optional GraphRAG behavior.

The suite will read evaluation cases from a test file (for example `test.json`), run end-to-end RAG answers, and score outputs with selectable judge frameworks (LLM-as-a-judge, RAGAS, FlashRAG, DeepEval, Bergen).

This phase is design only (no runtime implementation yet).

## 2. Required Benchmark Flow
The benchmark execution flow should be:
1. Read benchmark run config.
2. Build or load KB by calling existing service functions (same paths as UI/MCP).
3. Load test cases from `benchmark/datasets/test.json` (or configured path).
4. For each test case, call existing retrieval/generation service path to produce final RAG answer.
5. Send outputs to one or more configured evaluator frameworks.
6. Aggregate scores and produce run artifacts.

## 3. Scope and Non-Goals
### In scope
- Local Python script workflow under `benchmark/`.
- Strict reuse of current service-layer business logic (`RAG/services/*`) rather than separate benchmark-only RAG logic.
- Test set driven by query + ground truth records.
- Multiple evaluator frameworks selectable per run.
- Reproducible, inspectable run artifacts.

### Out of scope (for now)
- UI integration.
- MCP integration.
- CI scheduling and regression gating.

## 4. Proposed Directory Structure
```text
benchmark/
  DESIGN.md
  run_benchmark.py                  # future CLI entrypoint
  configs/
    default.yaml                    # run config (future)
  datasets/
    test.json                       # query + ground truth dataset (future)
  core/
    types.py                        # run config, test case, prediction, score schemas
    orchestrator.py                 # pipeline coordinator
    service_bridge.py               # wrappers to call RAG/services APIs
  evaluators/
    base.py                         # evaluator interface
    llm_judge.py                    # direct rubric LLM judge
    ragas_adapter.py                # RAGAS adapter
    flashrag_adapter.py             # FlashRAG adapter
    deepeval_adapter.py             # DeepEval adapter
    bergen_adapter.py               # Bergen adapter
  reports/
    aggregate.py                    # metrics/statistics aggregation
    render.py                       # markdown/json report rendering
  artifacts/
    <run_id>/
```

## 5. Service Reuse Contract
Benchmark code should call the same orchestration entrypoints as UI/MCP.

### Services to reuse
- `RAG/services/kb_service.py` for KB build/load/append behavior.
- `RAG/services/retrieval_service.py` for retrieval pipeline.
- `RAG/services/generation_service.py` for final answer generation.

### Design rule
- No duplicated benchmark-specific retrieval/generation implementation.
- Benchmark should pass configuration/dependencies into existing services and record outputs.
- If the benchmark needs additional observability (timings, retrieved chunks), add thin wrappers around service calls rather than forking logic.

## 6. Test Dataset Contract (`test.json`)
Initial dataset format should be explicit and simple.

### Minimum fields per test case
- `id`: unique identifier.
- `query`: user-style question.
- `ground_truth`: expected answer text (or canonical answer summary).

### Recommended optional fields
- `topic`: category for slicing.
- `difficulty`: easy/medium/hard.
- `must_include`: list of required facts/keywords.
- `metadata`: free-form extra annotations.

## 7. Evaluator Framework Selection
A run can choose one or multiple evaluators. All evaluators must map their outputs to a common normalized score schema for comparability.

### Target evaluator options
- `llm_judge` (custom rubric-based LLM-as-a-judge)
- `ragas`
- `flashrag`
- `deepeval`
- `bergen`

### Common normalized output keys
- `faithfulness`
- `answer_relevance`
- `context_relevance`
- `completeness`
- `overall`
- `justification` (optional)
- `raw_framework_output` (serialized, for audit/debug)

## 8. Configuration Model (Draft)
Benchmark runs should be fully config-driven.

### Required config sections
- `run`: name, seed, artifact path, logging level.
- `knowledge_base`: mode (`build` or `load`) and parameters passed to `kb_service`.
- `dataset`: path to `test.json`, optional sample size/filtering.
- `rag_execution`: parameters passed to retrieval/generation service calls.
- `evaluators`: list of evaluator framework configs.
- `aggregation`: summary metrics and optional per-slice metrics.

### Example evaluator config concept
- evaluator name/framework id
- model/provider settings (if relevant)
- framework-specific options
- metric normalization mapping

## 9. Artifact Contract (Per Run)
Store complete artifacts under `benchmark/artifacts/<run_id>/`.

- `config.snapshot.yaml`: exact effective config.
- `test_cases.snapshot.json`: exact evaluated cases.
- `predictions.jsonl`: per-case query, answer, and retrieval traces.
- `scores.<framework>.jsonl`: per-framework per-case scoring outputs.
- `scores.normalized.jsonl`: normalized scores across frameworks.
- `summary.json`: aggregated metrics overall and by slice.
- `comparison.csv`: framework-to-framework and/or system-to-system deltas.
- `report.md`: human-readable benchmark report.
- `run.log`: execution and error log.

## 10. Guardrails and Quality Controls
- Keep system identity hidden from judges where applicable.
- Use deterministic settings where possible (seed + low temperature for judge models).
- Capture raw evaluator outputs for post-hoc audits.
- Validate test dataset at load time and fail fast on schema errors.
- Record failures per case without silently dropping data.

## 11. Milestones
1. Define schemas and config for KB build/load + `test.json` evaluation.
2. Wire service-bridge design to call existing `RAG/services/*` entrypoints.
3. Add base evaluator interface and custom `llm_judge` design.
4. Add adapter designs for RAGAS, FlashRAG, DeepEval, Bergen.
5. Add normalization, aggregation, and report format.

## 12. Open Questions
- Which exact service function signatures should be treated as stable benchmark entrypoints?
- Should `ground_truth` support multiple accepted answers for one query?
- What is the normalization strategy when frameworks expose different metric sets?
- How should evaluator unavailability be handled (skip vs fail run)?
