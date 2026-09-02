# Threat model

Source: paper §Problem Formulation (`paper/behaviorguard.tex`), plus
`datasets/README.md` for corpus scope. This is a restatement, not a new claim.

## What is in scope

TurnShift (software; paper title still *BehaviorGuard*) considers an adversary
who has compromised a legitimate user account (account takeover, ATO) **or** who
pursues unauthorized access through a multi-turn social-engineering or
prompt-injection dialogue within an established session.

The adversary's goal is to issue high-impact requests—data exfiltration,
privilege escalation, authentication changes—through a conversational AI
interface that applies per-message content-based safety filtering.

The sequential study evaluates a **proxy**: mid-conversation **author
substitution** using real corpus messages from another user, not validated
real-world ATO incidents.

## What is out of scope

The paper does **not** model arbitrary open-ended LLM misuse (e.g. standalone
jailbreak prompts with no deviation from the user's accumulated behavioral
history). Empirical per-message evaluation is scoped to three injected anomaly
families: single-turn ATO bursts, progressive social-engineering escalations,
and prompt-injection turns embedded in otherwise benign multi-turn histories.

## Adversary capabilities (paper Table `tab:threat`)

| Capability | Assumed? | Notes |
|---|---|---|
| Send arbitrary text | yes | Fundamental capability |
| Control message timing | yes | Automated tooling available |
| Create multiple accounts | yes | No special access required |
| Observe anomaly scores | no | Not exposed in sidecar API |
| Know thresholds (τ1, τ2) | no | Security-by-obscurity |
| Know embedding model | no | Black-box assumption |
| Forge timestamps | no | Requires infra access |
| Inject into others' profiles | no | Requires backend write access |

## Non-adaptive adversary

The evaluation is a **non-adaptive** adversary who does not know the behavioral
profiling system exists and does not attempt to evade detection by mimicking
legitimate patterns. Adaptive adversaries (profile poisoning, threshold
probing, paraphrase evasion) are discussed as limitations, not as evaluated
attacks. Reported metrics are baseline figures against non-adaptive misuse, not
robustness guarantees against adaptive adversaries.

## Assumptions

1. Sufficient interaction history per user (≥ 10 prior messages for calibrated
   profiling; cold-start handling otherwise).
2. Each account corresponds to a single primary user.
3. Inputs are textual (no voice, keystroke timing, or device fingerprints).

## Corpora used as streams

PersonaChat, Blended Skill Talk, and Anthropic HH are public conversational
benchmarks (see `datasets/README.md`). Sequential-study positives are
author-substitution episodes built by `tools/build_ato_episode_dataset.py`,
not template injections.
