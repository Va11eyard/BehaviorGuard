# Final ablation table — PersonaChat corrected holdout (manuscript-ready)

**Verification:** All TP/FP/FN counts recomputed 2026-07-13 from `sling_audit_component_scores.npz`,
`mahalanobis_comparison_scores.npz`, and `threshold_sweep_cv_protocol.json`. Every cell matches.

**Sources:**
| Row | Primary artifact |
|-----|------------------|
| 1 | `sling_fix_confidence_intervals.json` + `sling_audit_component_scores.npz` |
| 2 | `sling_fix_confidence_intervals.json` + `mahalanobis_comparison_scores.npz` |
| 3–4 | `threshold_sweep_cv_protocol.json` (pooled test, fold-specific τ*) |

**Note on row 2 precision:** Point estimate **0.364** = 8/(8+14) from recomputed confusion at τ=0.60.
The JSON Clopper–Pearson precision point (0.381) assumed FP=13 in an earlier hardcoded constant;
recomputed FP=**14** is authoritative for this table.

---

## Markdown table

| Config | Protocol | τ | F1 [95% CI] | Precision | Recall | TP/FP/FN |
|--------|----------|---|-------------|-----------|--------|----------|
| s_ling included | Full holdout, fixed τ | 0.60 | 0.004 [0.000, 0.009] | — | 0.069 | 2/1078/27 |
| s_ling excluded | Full holdout, fixed τ | 0.60 | 0.314 [0.143, 0.474] | 0.364 | 0.276 | 8/14/21 |
| s_ling excluded (cosine) | 5-fold CV, pooled | 0.61* | 0.364 [0.170, 0.536] | 0.533 | 0.276 | 8/7/21 |
| Mahalanobis | 5-fold CV, pooled | 0.36* | 0.273 [0.095, 0.444] | 0.400 | 0.207 | 6/9/23 |

\*τ* uniform across all 5 folds (cosine: 0.61; Mahalanobis: 0.36). Selected by F1-max on each fold's 20-user validation set; pooled test counts sum across 5 fold test evaluations (29 positives total, 10,165 messages).

**Protocol column definitions:**
- **Full holdout, fixed τ** — Corrected PersonaChat, per-user 80/20 chronological split; N=10,165 held-out test messages, 29 positives; classification at fixed τ=0.60.
- **5-fold CV, pooled** — Same holdout pool; 25 holdout-positive users in 5 folds (seed=42); organic users hash-bucketed into folds; τ* per fold on validation users; pooled TP/FP/FN and bootstrap F1 CI on all fold test messages.

---

## LaTeX table (paste-ready)

```latex
\begin{table}[t]
\centering
\caption{PersonaChat corrected-holdout ablation: linguistic component and semantic scoring mode.
Rows~1--2 use the full 10{,}165-message test partition at fixed $\tau{=}0.60$.
Rows~3--4 use 5-fold user-level CV ($\tau^*$ selected per fold on validation users;
pooled test counts across folds; linguistic component excluded).
F1 intervals: bootstrap 95\% CI (message-level resampling, $n{=}5000$, seed~42).}
\label{tab:personachat-ablation}
\small
\begin{tabular}{llcccccc}
\toprule
Config & Protocol & $\tau$ & F1 [95\% CI] & Prec. & Rec. & TP/FP/FN \\
\midrule
$s_{\text{ling}}$ included
  & Full holdout, fixed $\tau$
  & 0.60
  & 0.004 [0.000, 0.009]
  & ---
  & 0.069
  & 2/1078/27 \\
$s_{\text{ling}}$ excluded
  & Full holdout, fixed $\tau$
  & 0.60
  & 0.314 [0.143, 0.474]
  & 0.364
  & 0.276
  & 8/14/21 \\
$s_{\text{ling}}$ excluded (cosine)
  & 5-fold CV, pooled
  & 0.61$^*$
  & 0.364 [0.170, 0.536]
  & 0.533
  & 0.276
  & 8/7/21 \\
Mahalanobis
  & 5-fold CV, pooled
  & 0.36$^*$
  & 0.273 [0.095, 0.444]
  & 0.400
  & 0.207
  & 6/9/23 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Limitations paragraph (paste-ready)

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
Headline metrics on this partition are reported as point estimates with 95\%
confidence intervals (bootstrap for F1; exact Clopper--Pearson for recall and
precision where reported); wide intervals reflect the small positive class.
Because the single-split validation half contained only 11 positives ($<12$),
threshold selection for semantic-mode comparison uses 5-fold user-level cross-validation
(Table~\ref{tab:personachat-ablation}, rows~3--4); fixed-$\tau{=}0.60$ full-holdout
rows (1--2) isolate the linguistic-component ablation.
Mahalanobis semantic scoring improves ROC-AUC directionally ($+0.019$ bootstrap
point estimate, 95\% CI includes zero) but does not improve pooled F1 under CV
relative to cosine (0.273 vs.\ 0.364).
Among the 29 held-out positives at $\tau{=}0.60$ (8 detected, 21 missed),
missed injections occupy a non-overlapping lower $s_{\text{sem}}$ region
(detected: $M{=}0.659$, $SD{=}0.056$, $n{=}8$; missed: $M{=}0.469$, $SD{=}0.025$,
$n{=}21$; pooled $SD{=}0.036$; Cohen's $d{=}5.31$), indicating a
\emph{representational} rather than threshold-placement limitation on semantic
scoring; composite near-miss mass does not yield a viable $\tau$ trade-off
(F1 falls from 0.314 at $\tau{=}0.60$ to 0.033 at $\tau{=}0.55$).
Expanding injection coverage and embedding capacity is explicit post-submission
work (BehaviorGuard v2).
```

---

## Verification log (automated)

```
R1 TP=2 FP=1078 FN=27  F1=0.003607  CI=[0.000,0.009]  rec=0.069  OK
R2 TP=8 FP=14  FN=21   F1=0.314     CI=[0.143,0.474]  prec=0.364 rec=0.276 OK
R3 TP=8 FP=7   FN=21   F1=0.364     CI=[0.170,0.536]  tau*=0.61 all folds OK
R4 TP=6 FP=9   FN=23   F1=0.273     CI=[0.095,0.444]  tau*=0.36 all folds OK
```
