# LTSMiTransformer: Long-Term Series Forecasting with Memory-Augmented Transformer

LTSMiTransformer is a lightweight yet powerful Transformer-based architecture designed for long-term time series forecasting (LSTF). It features two key innovations:

- **Learnable Temporal Sparse Attention (LTSA):** Dynamically selects and prunes irrelevant attention connections to reduce computational cost without compromising accuracy.
- **Memory-Augmented Module (MAM):** Integrates stable and volatile memory banks for long-term dependency retention and short-term adaptation.

---

## 🔧 Features

- Handles long sequence lengths (up to 10k steps)
- Efficient: 5.3× less GPU memory than FEDformer
- Accurate: Maintains or improves accuracy across 8 benchmark datasets
- Robust to noise and missing data
- Easy to extend for other sequence modeling tasks

---

## 📦 Installation

```bash
git clone https://github.com/your-username/LTSMiTransformer.git
cd LTSMiTransformer
pip install -r requirements.txt
