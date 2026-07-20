# MkDocs – Local Preview and gh-pages Deployment

## Local preview

Install the docs dependencies (once per environment):

```powershell
pip install -e ".[docs]"
```

> Note: the quotes are required in PowerShell to prevent `[docs]` from being
> interpreted as a wildcard.

Start the live-reloading dev server:

```powershell
mkdocs serve
```

Open <http://localhost:8000> in a browser. The page reloads automatically
whenever a source file listed under `watch:` in `mkdocs.yml` changes.

To do a one-off static build (output goes to `site/`):

```powershell
mkdocs build
```

---

## Deploying to GitHub Pages (gh-pages branch)

Docs are built and published automatically via the
`.github/workflows/docs.yml` workflow using
[mike](https://github.com/jimporter/mike) for versioned docs.

### Automatic triggers

| Event                                       | What gets deployed                                          |
| ------------------------------------------- | ----------------------------------------------------------- |
| Push / merge to `main`                      | Updates the **`dev`** alias                                 |
| Push of a version tag `v*` (e.g. `v0.17.0`) | Deploys that version **and** updates the **`latest`** alias |

The canonical URL (`https://sebi06.github.io/czitools/`) always points at
`latest`.

### Manual trigger (workflow_dispatch)

You can trigger the workflow manually from GitHub without pushing code:

1. Go to **Actions → docs** on GitHub.
2. Click **Run workflow** → select the `main` branch → **Run workflow**.

This is useful to refresh docs after editing docstrings without making a
dummy commit.

### Requirements

- The repo's GitHub Pages source must be set to the **`gh-pages` branch**
  (Settings → Pages → Source).
- The workflow writes to that branch using the `GITHUB_TOKEN` that is
  automatically provided; no extra secrets are needed.
