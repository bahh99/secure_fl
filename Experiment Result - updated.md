# Experimental Results Summary

This document provides a summary of gradient inversion attack tests conducted on Differential Privacy in Secure Federated
Averaging Algorithm on a Federated Learning Env, detailing scenarios both with and without defensive mechanisms.

## 1. Without Defenses
- **Reconstruction Quality:** High
- **MSE:** ~0.17
- **Interpretation:** The attacker successfully reconstructed input samples, indicating severe privacy risks in the absence of defenses.

## 2. With Differential Privacy Defenses (Gaussian Noise + Gradient Clipping)
- **Reconstruction Quality:** Very Poor
- **MSE:** ~5003.13
- **Interpretation:** Differential Privacy methods effectively protected against reconstruction attacks by significantly degrading reconstruction quality.

## 3. Anomaly Detection
- **Detection Result:** Successfully identified malicious client behavior (Client ID: 0).
- **Indicators:** Masking irregularities and invalid gradient submissions.
- **Detection Accuracy:** Highly accurate, aligning perfectly with the timing of attacks and defense activations.

## 4. Evaluation Summary
| Training Round | Attacker Active | Defenses            | MSE      | Interpretation                         |
|----------------|-----------------|---------------------|----------|----------------------------------------|
| 2              | Yes             | None                | 0.17     | Successful attack (True Positive)      |
| 3              | Yes             | Gaussian + Clipping | 5003.13  | Attack blocked effectively (True Negative) |

### Key Takeaways
- No false positives or negatives were recorded.
- Differential Privacy mechanisms significantly mitigated attack effectiveness.
- The anomaly detection mechanism reliably identified malicious behaviors.

These results emphasize the critical role of privacy-preserving strategies and anomaly detection mechanisms in securing federated learning environments.

