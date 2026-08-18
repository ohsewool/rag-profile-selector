# Retrieval profile vs citation quality (synthetic corpus)

Synthetic fixtures, not real documents: this measures whether the two kinds
of quality move together, not how any retriever performs in the world.

| profile | precision | recall | grounding | top-1 exact | page acc.\* | region acc.\* | misplaced | ungrounded |
|---|---|---|---|---|---|---|---|---|
| bm25-k4 | 0.5 | 1.0 | 0.875 | 0.75 | 0.625 | 0.625 | 3 | 1 |
| dense-k4 | 0.5 | 1.0 | 1.0 | 0.75 | 0.5 | 0.5 | 4 | 0 |
| hybrid-rrf-k4 | 0.5 | 1.0 | 0.875 | 1.0 | 0.625 | 0.625 | 3 | 1 |
| hybrid-rrf-k8 | 0.5 | 1.0 | 0.875 | 1.0 | 0.625 | 0.625 | 3 | 1 |

\* averaged over every returned citation, so an ungrounded result raises the
  average by leaving the denominator; read `top-1 exact` for the unbiased view.


## Reading

- retrieval recall spread across profiles: 0.0
- citation top-1 exact spread across profiles: 0.25
- citation quality separates the profiles **more** than retrieval quality does,
  so a selector tuned on recall alone would be choosing blind on this axis.
