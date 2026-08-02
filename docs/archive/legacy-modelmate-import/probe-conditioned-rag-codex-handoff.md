# Codex 인수인계 프롬프트

아래 프롬프트를 새 Codex 작업에 그대로 전달한다.

---

You are working on a research implementation titled:

**Probe-Conditioned RAG Retrieval Profile Selection**  
**Probe 검색 신호 기반 질의별 RAG 검색 프로필 선택**

Do not modify code yet.

## Phase 0: Repository inspection and experimental feasibility

Before any code modification:

1. Read all applicable repository instructions, including:
   - root `AGENTS.md`
   - `.codex/AGENTS.md`
   - any nested `AGENTS.md`
2. Read:
   - `docs/probe-conditioned-rag-research-spec.md`
   - the existing README and dependency manifests
3. Inspect the actual repository tree, runtime, tests, storage, deployment files, and current product boundaries.
4. Confirm whether this repository is still the ModelMate CSV AutoML SaaS repository.
5. Identify any conflict between the existing product rules and the new RAG research.
6. Do not alter the existing frontend, FastAPI routes, CSV AutoML pipeline, deployment configuration, or user data.
7. Assess whether implementation should live:
   - in a separate research repository, or
   - in an isolated `research/probe_conditioned_rag/` package.
8. Estimate:
   - HotpotQA and SciFact download size
   - processed corpus/index size
   - BM25 indexing requirements
   - dense embedding/index requirements
   - CPU/GPU/RAM/storage requirements
   - expected profile-run count
   - expected runtime for the fixed MVP cohort
9. Check whether required tools can be supported without destabilizing the current app:
   - Python version
   - Java/Pyserini compatibility
   - FAISS availability
   - `sentence-transformers`
   - `scikit-learn`
   - Parquet support
10. Check dataset licenses and redistribution constraints from official sources.

## Required Phase 0 output

Report before editing:

1. Repository findings
2. Existing constraints that apply
3. Feasibility result
4. Recommended isolation strategy
5. Proposed exact file tree
6. Dependency impact
7. Compute/storage estimate
8. Risks and blockers
9. Small first implementation milestone
10. Verification plan

If implementation in the current repository would violate existing `AGENTS.md` or risk the production app, stop and recommend a separate repository. Do not bypass repository rules.

## Approved research scope

Treat `docs/probe-conditioned-rag-research-spec.md` as the source of truth.

Core:

- HotpotQA MVP
- HotpotQA + SciFact final
- four pilot retrieval profiles
- conditional expansion to six core profiles
- optional two reranker profiles only after the validation gate
- one configuration-conditioned evidence utility predictor
- query-only vs query-plus-probe comparison
- gold evidence retrieval quality as the training label
- deterministic cost calculation
- measured latency median and p95
- one validation-selected budget
- fixed, rule-based, query-only, query-plus-probe, and oracle baselines

Do not add:

- domain routing
- cost prediction
- latency prediction
- answer generation for every training query/profile
- MVP verifier/retry/abstention
- LLM verifier
- query decomposition
- complex rewriting
- multiple retries
- three separate operating modes
- history conditioning in the primary experiment
- QASPER before primary experiments are complete
- a UI, API, SaaS integration, or autonomous agent

## Research integrity rules

- Do not claim this is the first general RAG Auto-Tuner.
- Do not claim a new general framework.
- Positive and negative findings are equally valid.
- Do not tune on test data.
- Do not leak gold answer/evidence into query or probe features.
- Do not change profile gates after seeing test results.
- Do not fabricate metrics, traces, retrieval results, costs, or benchmark outputs.
- Record seeds, hashes, model revisions, package versions, and split manifests.
- Keep raw data, indexes, and large artifacts out of Git.
- Core experiments must not require an LLM API key.

## Implementation protocol after approval

After Phase 0 is reviewed and implementation location is approved:

1. Implement only the smallest next milestone.
2. Start with dataset schema validation, deterministic split manifests, and leakage tests.
3. Add retrieval profiles only after the data protocol passes.
4. Run the smallest relevant tests after each milestone.
5. Record commands, runtime, artifact paths, and failures honestly.
6. Do not begin SciFact, rerankers, answer generation, history ablation, or retry logic during the HotpotQA data/index milestone.
7. Stop if official data cannot be obtained, licenses are unclear, required compute is unavailable, or the existing repository rules prohibit the change.

## Initial acceptance criteria

Phase 0 is complete only when:

- repository compatibility is documented
- implementation location is justified
- dependency and compute feasibility are estimated
- official dataset sources are identified
- no code has been modified
- the first implementation milestone is narrowly scoped
- a verification plan exists

Begin with repository inspection and experimental feasibility analysis. Do not modify any file until that report is complete.

