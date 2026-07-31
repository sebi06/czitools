---
name: token-anxiety-manager
description: "Estimate the cost of a request before acting, suggest cheaper alternatives, and optionally track usage in a local budget file."
argument-hint: "(optional) 'config' to edit budget settings, 'status' for the current period report, 'reset' to reset the period, or leave blank for the normal pre-flight check"
---

# Token Anxiety Manager

Use this skill before expensive or ambiguous work to estimate cost, compare model tiers, and surface cheaper manual offloads. It is intentionally generic so you can copy it into another repository and adjust the local budget settings.

## Activation

Invoke this skill:
- At the start of every user request.
- Before a broad search, a large file read, a multi-file refactor, or a subagent call.
- Before an operation described as audit, review-all, find-all, or cross-cutting.

## Triage

Classify the request in one short paragraph.

| Class | Examples | Action |
|-------|----------|--------|
| `trivial` | typo fix, rename, single-line edit, factual answer already in context | Proceed with the cheapest model tier. |
| `standard` | edit 1–3 files, write a small test, inspect a couple of files and patch | Run the estimation step. |
| `complex` | multi-file change, non-trivial feature, intricate debugging | Run the estimation step and recommend a standard model by default. |
| `massive` | repo-wide audit, broad refactor, large generation task | Estimate, then suggest breaking the task into smaller chunks. |

## Estimation

Use conservative heuristics.

| Operation | Input tokens | Output tokens |
|-----------|--------------|---------------|
| Read a file | `file_bytes / 4` | ~50 |
| Anchored search | ~500 | `matches × 100` |
| Broad search | ~2,000 | `matches × 100` |
| Short shell command | ~200 | `stdout_bytes / 4` |
| Long shell command | ~200 | ~3,000 |
| Edit | ~`diff_bytes × 2` | ~`diff_bytes` |
| Write new file | ~200 | `content_bytes / 4` |
| Spawn agent | ~5,000 | ~10,000–50,000 |
| Web fetch | ~500 | `page_bytes / 4` |
| Reasoning + final reply | ~500 | ~500–2,000 |

Estimate raw tokens as input + output. If you track usage units, convert with your own multiplier or budget formula. Keep the math simple and clearly label it as an estimate.

## Recommend

Use model tiers rather than hard-coded model names.

- `trivial` -> cheap tier
- `standard` -> standard tier
- `complex` -> standard tier by default; premium only when the task truly needs deeper reasoning
- `massive` -> break the task into smaller chunks before choosing a model

If a more expensive tier is chosen than necessary, tell the user what they could save by downshifting.

## Manual offloads

For standard or larger tasks, list 1–3 concrete offloads that reduce the estimate.

- Paste the relevant file section instead of having me read the whole file.
- Paste the exact error output instead of re-running a failing command.
- Use your editor's rename or refactor tools, then ask me to review the diff.
- Run the build or test locally and paste only the failures.
- Narrow the search to a specific folder or file.
- Skip verification if you already ran it and can share the result.

## Present

Output exactly this block:

```md
## Token Anxiety Check
- Task: <class> — <one-line description>
- Current model: <model> (multiplier: Nx)
- Estimated: ~X budget units (~Y raw tokens)
- Budget: U / B used this period (P% — Q remaining)
- Recommendation: <current is fine | downshift to <model>>
- Manual offloads available:
  - <offload 1>
  - <offload 2>

Proceed on <model>, downshift, or want to offload? (proceed / downshift / manual)
```

For `trivial`, collapse to one line:

`Token Anxiety: trivial on <model> (~0 budget units). Proceeding.`

## Budget tracking

If you want local tracking, store it in a JSON file such as `.claude/token-budget.json` and update it at the end of each request. Keep the schema small and editable.

Example:

```json
{
  "monthlyBudgetUnits": 1500,
  "currentPeriod": "YYYY-MM",
  "usedBudgetUnits": 0,
  "models": {
    "cheap-model": { "multiplier": 0, "tier": "cheap" },
    "standard-model": { "multiplier": 1, "tier": "standard" },
    "premium-model": { "multiplier": 10, "tier": "premium" }
  },
  "warnAt": [0.5, 0.8, 0.95]
}
```

If the period changes, stop and confirm before resetting counters. Never reset silently.

## Rules

- Be honest about uncertainty.
- Keep the check short.
- Prefer cheaper options when the task is simple.
- Do not pad the output.
- Use local paths and settings only if they make sense for the repository where the skill is installed.
