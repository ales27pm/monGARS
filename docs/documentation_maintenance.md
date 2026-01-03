# Documentation Maintenance Checklist

> **Last updated:** 2025-11-25 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This checklist keeps every Markdown guide in the repository dynamic. Follow the
steps whenever you touch project documentation so contributors always land on
fresh instructions that match the shipped code.

## Quick reference
| Action | Where to record it | Owner |
| --- | --- | --- |
| Add or update a runbook | Link it from [docs/index.md](index.md) and note the change in the pull request summary | Author |
| Update architecture diagrams | Regenerate the asset under `docs/images/` and confirm the README renders it | Author |
| Modify CI or release workflows | Cross-link [docs/workflow_reference.md](workflow_reference.md) and capture new commands | Author + reviewer |
| Change SDK behaviour | Update [docs/sdk-overview.md](sdk-overview.md) and [docs/sdk-release-guide.md](sdk-release-guide.md) | SDK maintainer |
| Touch AGENTS guidance | Run `python scripts/manage_agents.py refresh` so scoped rules stay in sync | Author |

## Update workflow
1. **Track scope** – List every affected guide in your pull request description
 and mention the owner who should review the change (ops, SDK, research, etc.).
2. **Refresh the banner and charters** – Run `python scripts/update_docs_metadata.py`
   to sync the `Last updated` line for every Markdown or MDX file you touched. The
   script derives timestamps from Git history and removes stale manual banners.
   Follow it with `python scripts/manage_agents.py refresh` whenever you edit
   scoped AGENTS files or their config. CI reruns both helpers in the
   `docs-metadata` workflow job after changes land; if the rerun finds drift it
   uploads a `docs_metadata.patch` artifact covering banners and AGENTS text,
   then lists the impacted files in the job summary so you can apply the fix
   locally with `git apply docs_metadata.patch`.
3. **Validate links** – Run `npx markdownlint-cli@0.39.0 "docs/**/*.md" "README.md"` to enforce shared formatting rules and
   follow it with `npx markdown-link-check -q README.md` plus `npx markdown-link-check -q docs/index.md` for external link verification.
4. **Keep commands accurate** – Execute the documented command locally or paste
   the latest CI output. Replace stale timings, flags, or expected results.
5. **Confirm hub coverage** – Ensure [docs/index.md](index.md) references the
   guide you touched. If the doc moves directories, update every inbound link.
6. **Call out deltas** – Summarise material changes in the README or release
   notes so operators know a runbook changed before the next deploy window.

## Review tips
- Reviewers skim this file first to confirm the checklist was followed; reference
  the bullet numbers when requesting fixes (e.g. "fails checklist step 4").
- Encourage incremental updates—merging partial improvements is preferable to
  waiting for a perfect rewrite, provided the banners and hub links stay current.
- When in doubt, prefer linking to the source file or script rather than copying
  large code snippets that may drift as APIs evolve.
