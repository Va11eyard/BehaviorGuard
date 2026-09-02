# Original (confounded) vs Corrected pipeline evaluation

| Dataset | Method | Orig F1 | Corr F1 | Δ F1 | Orig R | Corr R | Substantial drop? |
|---------|--------|---------|---------|------|--------|--------|-------------------|
| PersonaChat | behaviorguard | 0.6933 | 0.2069 | -0.4864 | 1.0 | 1.0 | YES |
| PersonaChat | isolation_forest | 0.7586 | 0.4444 | -0.3142 | 0.8462 | 0.6667 | YES |
| PersonaChat | autoencoder | 0.9455 | 0.75 | -0.1955 | 1.0 | 1.0 | YES |
| BST | behaviorguard | 0.7939 | 0.3656 | -0.4283 | 1.0 | 0.8947 | YES |
| BST | isolation_forest | 0.7786 | 0.3673 | -0.4113 | 0.9808 | 0.4737 | YES |
| BST | autoencoder | 0.782 | 0.6061 | -0.1759 | 1.0 | 0.5263 | YES |
| AnthropicHH | behaviorguard | 0.6424 | 0.2655 | -0.3769 | 0.9298 | 1.0 | YES |
| AnthropicHH | isolation_forest | 0.5714 | 0.3077 | -0.2637 | 0.5965 | 0.2667 | YES |
| AnthropicHH | autoencoder | 0.6826 | 0.2745 | -0.4081 | 1.0 | 0.4667 | YES |
