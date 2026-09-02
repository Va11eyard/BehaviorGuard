# Limitations

Sources: paper §`sec:cusum:limitations` and §`sec:limitations`
(`paper/behaviorguard.tex`). All items below are already documented there.
They are not softened here.

## Sequential study (primary claim)

1. **Persona-conditioning.** Absolute numbers on persona-conditioned corpora
   overstate what unconstrained traffic will yield. PersonaChat is the
   upper-bound dataset; BST is the more honest headline (AUC 0.900 / 35.0%
   detection at FA = 1/1,000).

2. **Third corpus is Anthropic HH, not WildChat-1M.** WildChat/ShareGPT
   downloads stalled. HH lacks cross-session hashed identity, so streams are
   conversation-scoped. HH organic streams: AUC 0.795 / 20.0% detection at
   FA = 1/1,000 — a lower-bound estimate.

3. **Stylometric residuals underperform** at conversational message length
   (≈13 tokens per message on BST). Style markers are too sparse per message
   to accumulate reliably, which is why the embedding residual dominates and
   why `cusum_combined` (diluting embedding signal with stylometric noise)
   underperforms `cusum_embed`.

4. **Mimicry / nearest-centroid donor selection.** Random donor selection
   overstates detectability against a mimicry attacker: nearest-centroid donor
   episodes drop PersonaChat FA = 1/1,000 detection from **63% to 35%**
   (AUC 0.870; `results/primary/sequential_ato_study_mimicry.json`).

5. **Online profile update during the attack.** An online profile that absorbs
   each stream message after scoring loses about **8 detection points**
   relative to a frozen profile (71.7% → 63.7% in the lightweight residual
   protocol; `results/primary/sequential_ato_online_dual_personachat.json`).

6. **Dual-λ divergence CUSUM failed as a poisoning defense.**
   λ_fast = 0.5, λ_slow = 0.99: AUC ≈ 0.55, FA1 ≈ 3.7%. Reported as a
   **negative result**, not as a fix.

## Broader system limitations

7. **Cold start / coverage gap.** Calibrated profiling wants ≥ 10 prior
   messages. On PersonaChat, **65% of evaluated test users have fewer than 10
   prior training messages**. New accounts fall back to content-only checks,
   eliminating the behavioral advantage.

8. **Text-only.** Voice characteristics, keystroke timing, and device
   fingerprints are excluded. Shared accounts and organizational access are
   out of scope.

9. **Adaptive evasion is unevaluated.** Paraphrase attacks, threshold probing,
   and knowledgeable profile poisoning are explicit out-of-scope threats.
   Operation-risk tiers use substring matching without a secondary classifier.

10. **Embedding model scope.** All results use `all-MiniLM-L6-v2`
    (384-dimensional). Performance under larger or domain-specific encoders
    has not been evaluated.

11. **Per-message F1 on positive-enriched samples is not a supported claim.**
    The archived Table III protocol (~48% prevalence) reported F1 of 0.693 /
    0.794 / 0.642. At the full holdout's 0.29% prevalence the identical
    configuration collapses to F1 = 0.0059 with a 90.5% false-positive rate.
    See `results/archived-per-message-study/`.
