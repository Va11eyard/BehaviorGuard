# s_ling FIX_IT — Scope and Limitations (ready-to-paste)

**Target placement:** `\subsection{Limitations}` (new or expanded) in `paper/behaviorguard.tex`; footnote on Table~\ref{tab:main} (`tab:main`) and any corrected-dataset diagnostic table reporting F1/recall/AUC on PersonaChat.

---

## Manuscript paragraph (LaTeX)

```latex
\paragraph{Evaluation scope and statistical power.}
Our corrected PersonaChat evaluation injects synthetic adversarial turns into
35 of 5{,}000 user timelines (0.7\% user coverage), yielding 126 injected
messages in the full dataset and 29 positive test messages under the standard
80/20 per-user holdout (10{,}165 test messages total).
These results support \emph{proof-of-concept} behavioral detection---demonstrating
that per-user profiling can flag injected account-takeover, social-engineering,
and prompt-injection templates---but they do \emph{not} establish statistical
significance or broad generalization to arbitrary anomalous behavior.
Headline metrics on this partition are therefore reported as point estimates
accompanied by 95\% confidence intervals---bootstrap for F1, exact
Clopper--Pearson for recall and precision (Table~\ref{tab:main},
footnote~\footnotemark); wide intervals reflect the small positive class, not
hidden variance.
Expanding injection coverage across users and anomaly families is explicit
post-submission work (BehaviorGuard v2), not a limitation we omit from the
current scope statement.
\footnotetext{Corrected PersonaChat holdout at $\tau_2{=}0.60$ with linguistic
component excluded: F1 $= 0.314$ [0.143, 0.474] (bootstrap),
recall $= 0.276$ [0.127, 0.472], precision $= 0.381$ [0.181, 0.616]
(Clopper--Pearson); with linguistic included: F1 $= 0.004$ [0.000, 0.009]
(bootstrap); see \texttt{sling\_fix\_confidence\_intervals.json}.}
```

---

## Plain-text summary (for internal review)

- **Coverage:** 35 / 5,000 users (0.7%), 126 total injected messages, 29 test positives.
- **Claim scope:** Proof-of-concept on three template families (ATO, social engineering, prompt injection)—not arbitrary anomaly detection.
- **Statistics:** F1 uses bootstrap 95% CI (message-level resampling, n=5000); recall/precision use exact 95% Clopper–Pearson CIs; do not claim significance or generalization.
- **Future work:** Dataset coverage expansion is v2 / post-submission, stated explicitly.
- **Table III footnote:** Attach CIs to F1/recall/precision wherever corrected PersonaChat numbers appear; paper Table III uses a different protocol (20 sampled users, original dataset) and should cross-reference limitations if comparing.

---

## Pre-commitment: Mahalanobis reporting standard

Before Mahalanobis results are written into the manuscript:

1. **Point metrics** (F1, recall, precision, AUC) on the same 80/20 corrected PersonaChat test partition as the s_ling audit, at $\tau=0.60$, with **bootstrap 95% CI on F1** and **exact Clopper–Pearson CIs on recall/precision** (same method as `scripts/compute_sling_fix_confidence_intervals.py`).

2. **Cosine vs Mahalanobis comparison:** Bootstrap 95% CI on **AUC difference** (Mahalanobis − cosine), reusing the resampling logic in `production_sling_audit_snippet.py` (seed=42, n_bootstrap=5000). An improvement is described as **directional** unless the CI excludes zero. **Do not** claim a confirmed Mahalanobis improvement if the CI overlaps zero, regardless of the point estimate.

3. **Composite config for comparability:** `linguistic_component_enabled=False`, `overrides_enabled=False`, `lambda=0.50`, medium sensitivity—matching the post–s_ling-fix evaluation path unless explicitly noted otherwise.
