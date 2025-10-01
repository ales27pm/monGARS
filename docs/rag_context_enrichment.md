# RAG Context Enrichment

## Overview

RAG Context Enrichment enhances AI-assisted reviews by retrieving contextually relevant code patterns from indexed repositories. The feature augments pull request (PR) analysis with references to similar implementations, enabling reviewers to make faster, more informed decisions.

## Prerequisites

- Available exclusively to Qodo Enterprise single-tenant or on-premises deployments.
- Requires a prepared database and fully indexed codebase. Contact support if indexing has not been completed.

## Configuration

Add the following section to your configuration file to enable repository enrichment:

```
[rag_arguments]
enable_rag=true
```

### RAG Arguments

| Option          | Description |
| --------------- | ----------- |
| `enable_rag`    | Enable or disable repository enrichment. Defaults to `false`. |
| `rag_repo_list` | Repositories used for semantic search. Use `['all']` to search the entire codebase or specify a list such as `['my-org/my-repo']`. Defaults to the repository where the PR originated. |

## Applications

RAG context is surfaced in the following tools:

- `/ask`
- `/compliance`
- `/implement`
- `/review`

Within `/review`, the Focus area highlights findings derived from RAG data, and a dedicated References section lists every relevant source discovered during analysis.

## Limitations

- Natural-language search drives retrieval quality; results may vary between queries.
- To minimize noise, scope searches to the PR repository whenever possible.
- Requires a secure, private, and indexed codebase.
- Only available on Qodo Merge Enterprise single-tenant or on-premises deployments.

## Recommended Practices

- Keep repository indexing up to date to maintain high-quality results.
- Review and curate the `rag_repo_list` to balance context coverage against retrieval noise.
- Document any new configuration in deployment runbooks so operators can monitor and maintain the feature.
