# Status

## Current phase

**Approved specification and planning documentation.** The repository contains the controlled scaffold and the bounded probe-conditioned RAG study design. Application implementation, data/model downloads, index construction, retrieval runs, feature extraction, selector training, and experiments have not started.

## Completed

- Repository initialized on `main`.
- HotpotQA-only MVP, four pilot profiles, configuration-conditioned prediction design, feature groups, labels, baselines, split rules, metrics, exclusions, and completion criteria documented.
- Core scope, conditional final extensions, and rejected features separated.
- Leakage controls, profile stage gates, exact regret, deterministic cost, measured latency, and paired-bootstrap protocol documented.
- Planning-only `AUTO_READY` tasks and approval boundaries defined.

## Not started

- Repository inspection and environment bootstrap planning.
- HotpotQA manifest/evidence-map planning or validation.
- Dependency, sparse/dense retriever, tokenizer, model, or index selection.
- Application or test implementation.
- Dataset/model downloads or index construction.
- Profile runs, feature extraction, selector training, answer generation, or evaluation.
- Stage 2, SciFact, QASPER, reranking, history ablation, verifier, retry/escalation, or abstention.

## Safety and artifact state

- No datasets, corpora, models, or weights have been downloaded.
- No indexes, features, learned models, or result artifacts have been created.
- No APIs, dashboards, Docker, or MLflow have been added.
- No retrieval experiment or long-running job has been started.

## Next authorized work

Start with A1 in `docs/TASKS.md`: inspect and inventory the repository. This is a bounded planning task and does not authorize implementation or downloads.
