---
name: review-python-code
description: "Review Python code for quality, readability, correctness, and project conventions. Use when: code review, review this file, check code quality, Python review, review PR changes, review my code."
---

# Python Code Review — SegmentationService

Quick-checklist review for Python code in this repository. Scan the provided code against each section below and report findings grouped by category. Skip categories with no findings.

## Procedure

1. **Read** the file(s) or selection the user provides.
2. **Run** through each checklist section below.
3. **Report** findings as a markdown table: `| Line(s) | Category | Finding | Suggestion |`.
4. **Summarize** with a one-line verdict: LGTM, minor issues, or needs rework.

---

## Checklist

### 1 — Correctness & Logic
- Off-by-one, wrong comparison, unreachable code, swallowed exceptions.
- Mutable default arguments (`def f(x=[])`).
- Resource leaks (open files, DB sessions, Dask futures not released).
- Correct use of `db_session` inside Flask context; use `SessionHelper` outside it.
- Task state transitions follow `TaskState` enum (`ready → running → completed | error`).

### 2 — Security (OWASP Top 10)
- No unsanitized user input in SQL, shell commands, or file paths.
- No hardcoded secrets, tokens, or credentials.
- No `eval()`, `exec()`, `pickle.loads()` on untrusted data.
- Log files: no PII or secrets written via `lg.get_log()`.

### 3 — Project Conventions
- **Logging**: use `lg.get_log(__name__)` with `LogMessage` enum values — never `print()`.
- **Config**: access via `cfg.current_config.<ATTR>` — never hardcode paths or env vars.
- **Errors**: raise `SegmentationError`; format with `create_error_response(ex)` in views.
- **API responses**: business endpoints use `create_response()` (JSON:API v1.0). Operational endpoints (`/health`, `/gpu`, `/logs`) use `flask.jsonify()`.
- **ONNX-first**: use `deep_extractors_onnx.py` — never import from archived `deep_extractors.py`. `DnnClassifier` tries `OnnxInferencer` first; TF is lazy-imported fallback only.
- **Singletons**: `FeatureExtractorFactory`, `Scheduler`, `FutureStore` use `@Singleton`.
- **Type hints**: required on all public function signatures.
- **Import order**: `E402` is suppressed only in `runserver.py`, `runscheduler.py`, `scripts/`, `tests/` — everywhere else imports must be at the top.
- **TF isolation**: TensorFlow imports must be lazy (`import tensorflow as tf` inside functions) to preserve ONNX-only deployability.
- **GPU detection**: use `onnxruntime.get_available_providers()`, never TF-based detection.

### 4 — Readability & Style
- PEP 8 compliance (Ruff rules: E4, E7, E9, F; E501 not enforced).
- Meaningful names; no single-letter variables outside tight loops.
- Functions ≤ 50 lines; classes have a single clear responsibility.
- No dead code, commented-out blocks, or TODO/FIXME without a tracking reference.

### 5 — Performance & Resource Usage
- NumPy/OpenCV vectorized ops instead of Python loops on image data.
- ONNX sessions cached via `@lru_cache` — no redundant session creation.
- Dask futures properly tracked in `FutureStore` and released after use.
- Avoid loading large arrays into memory when tiling/streaming is available.

### 6 — Testability
- Pure functions preferred; side effects isolated behind interfaces.
- New logic should be testable without GPU (`IS_GPU` flag, CPU fallback).
- Mocking guidance: use `mock_helpers.py` patterns; mock at the boundary, not deep internals.
- keep in in mind that the test also need to be able to run in a CPU-only environment, so any GPU-specific code should be abstracted behind interfaces that can be mocked or have CPU fallbacks.

### Correctness of Doc

- check for a README.md file in the repository root that provides an overview of the project, installation instructions, usage examples, and contribution guidelines.
- ensure that the README.md file is well-structured, easy to read, and free of spelling and grammatical errors.
- check for its internal consistency, ensuring that the information provided is accurate and up-to-date with the codebase and the project's current state.
