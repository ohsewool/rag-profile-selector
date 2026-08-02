# Tasks

Only the planning and bootstrap tasks under `AUTO_READY` may proceed without further approval. They do not authorize implementation, downloads, indexing, training, retrieval runs, or experiments.

## AUTO_READY

### Repository inspection

- [ ] **A1 — Inspect and inventory the repository.** Confirm branch and working-tree state; inventory documentation and configuration; verify that no application code, datasets, corpora, indexes, model artifacts, credentials, results, or unexpected dependencies are present. Record findings in `docs/STATUS.md`.
- [ ] **A2 — Review repository controls.** Check `.gitignore`, repository instructions, artifact boundaries, and documentation links against the approved scope, leakage controls, and reproducibility requirements.

### Environment bootstrap

- [ ] **A3 — Specify the minimal local Python environment.** Document a candidate Python version, isolated-environment procedure, shared cache locations, deterministic seed conventions, and verification commands. Do not install dependencies or download artifacts.
- [ ] **A4 — Plan local artifact boundaries.** Define locations and naming for manifests, corpus snapshots, indexes, probe caches, profile outputs, features, model artifacts, latency observations, and results. Keep datasets, models, indexes, and generated outputs out of Git.

### Data-protocol planning

- [ ] **A5 — Draft the HotpotQA manifest protocol.** Specify dataset version/source fields, corpus and query checksums, query IDs, evidence mapping validation, license/usage notes, and split-manifest schema without downloading data.
- [ ] **A6 — Draft the grouped split protocol.** Define query-group assignment, fixed seeds, manifest validation, and assertions that all profile rows for a query remain in one split and test outcomes cannot enter development.
- [ ] **A7 — Draft evidence-label definitions.** Specify Evidence F1, supporting-fact recall, and recall-threshold candidate targets; identify edge cases, missing evidence behavior, aggregation, and the approval point for freezing the primary target.
- [ ] **A8 — Draft feature contracts.** Define each approved query and probe feature, inputs, normalization scope, missing-value behavior, versioning, and leakage checks. Do not compute features.

### Profile-specification planning

- [ ] **A9 — Specify the four pilot profiles.** Document BM25 `k=4`, dense `k=4`, hybrid RRF `k=4`, and hybrid RRF `k=8`, including evidence unit, index contract, score/rank semantics, RRF formula, deduplication, and deterministic tie handling. Do not select dependencies/models or build indexes.
- [ ] **A10 — Draft the profile-diversity gate.** Define validation-only statistics and a proposed predeclared threshold for deciding whether Stage 2 is scientifically justified. Profile expansion remains in `NEEDS_APPROVAL`.
- [ ] **A11 — Draft deterministic cost and latency protocols.** Specify token/call accounting, fixed-price manifest versioning, warm-up/cache policy, timing scope, repetitions, hardware metadata, and median/p95 reporting. Do not measure retrieval.

### Evaluation-runner planning

- [ ] **A12 — Define evaluation data contracts.** Specify immutable query-profile result rows, selector inputs/outputs, profile descriptors, predictions, gold outcomes, selected profiles, oracle profiles, regret, latency, cost, and version metadata. Do not create executable code.
- [ ] **A13 — Define baseline protocols.** Specify validation-only construction for best fixed and rule-based selectors, matched query-only/query-plus-probe training, and the test-only oracle comparison.
- [ ] **A14 — Define metric and confidence-interval protocols.** Freeze planning definitions for retrieval metrics, exact regret, profile distributions, context tokens, Pareto analysis, and query-paired bootstrap 95% intervals, including a recorded resampling seed.
- [ ] **A15 — Draft the run manifest and stage gates.** Specify preflight checks, ordered stages, stop conditions, artifact checksums, and the approval checkpoints before any index build, retrieval run, training, Stage 2 expansion, or optional extension.

### Test planning

- [ ] **A16 — Draft data-leakage tests.** Plan assertions for grouped splits, manifest immutability, feature fitting boundaries, test isolation, and profile-row completeness.
- [ ] **A17 — Draft profile and metric tests.** Plan synthetic fixtures for RRF, cutoffs, deduplication, tie breaking, evidence metrics, regret, deterministic cost, Pareto classification, and paired bootstrap behavior without running them.
- [ ] **A18 — Draft reproducibility tests.** Plan checks for fixed seeds, version identifiers, stable feature extraction, raw result reconstruction, and repeated-run manifest equivalence.

## NEEDS_APPROVAL

- [ ] Begin any application implementation or create executable retrieval, feature, training, evaluation, or test code.
- [ ] Select, install, or add a dependency, retrieval/index library, sparse/dense model, tokenizer, framework, or model weight.
- [ ] Download HotpotQA, another dataset, a corpus, an index, or a model artifact.
- [ ] Build an index, compute features, run retrieval, train a selector, generate answers, or execute experiments.
- [ ] Start any long-running or heavy job.
- [ ] Expand beyond the four pilot profiles, including Stage 2 `k=8` profiles or Stage 3 rerankers.
- [ ] Add SciFact or QASPER.
- [ ] Add a verifier, retry, escalation, abstention, history feature, rewriting, decomposition, routing, or another capability.
- [ ] Freeze the primary target, minimum-recall threshold, meaningful-diversity threshold, deterministic profile tie-break, or numerical success/non-inferiority threshold.
- [ ] Add any feature, baseline, metric, dependency, paid API, API service, dashboard, Docker, MLflow, or scope change.

## BLOCKED

- None. Implementation, downloads, and experiments are approval-gated rather than current planning blockers.

## DONE

- [x] Initialize the controlled research repository on `main`.
- [x] Approve the probe-conditioned RAG research question, HotpotQA MVP, pilot profiles, predictor design, feature groups, baselines, evaluation, exclusions, and completion criteria.
- [x] Replace placeholder documentation with the approved specification and planning controls.
