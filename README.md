# LTSMiTransformer: Learnable Temporal Sparsity for Long-Term Time-Series Forecasting
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

LTSMiTransformer is a lightweight yet powerful Transformer-based architecture designed for long-term time series forecasting (LSTF). It features two key innovations:
    <p align="left">
    <img src=".\img\LTSMiTransformer_model_architecture.png" height = "360" alt="" align=center />
    <br><br>
    <b>Figure 1.</b> The LTSMiTransformer architecture.
    </p>
- **Learnable Temporal Sparse Attention (LTSA):** Dynamically selects and prunes irrelevant attention connections to reduce computational cost without compromising accuracy.
    
    <br>
    <p align="left">
    <img src=".\img\Sparsity factor.png" height = "320" alt="" align=center />
    <br><br>
    <b>Figure 2.</b> The sparsity factor.
    </p><br>
- **Memory-Augmented Module (MAM):** Integrates stable and volatile memory banks for long-term dependency retention and short-term adaptation.

## Features

- Handles long sequence lengths (up to 10k steps)
- Efficient: 5.3× less GPU memory than FEDformer
- Accurate: Maintains or improves accuracy across 8 benchmark datasets
- Robust to noise and missing data
- Easy to extend for other sequence modeling tasks

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

## Installation

```bash
git clone https://github.com/ypolycarp/LTSMiTransformer.git
cd LTSMiTransformer
pip install -r requirements.txt
```

## Usage
Commands for training and testing the model with *ProbSparse* self-attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

# ETTh2
python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

# ETTm1
python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
```

More parameter information please refer to `main_informer.py`.

## <span id="resultslink">Results</span>

We have provided the initial experiment results of forecasting on different datasets for different forecasting horizons.

<p align="center">
<img src="./img/Experimental Results.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 3.</b> Multivariate forecasting results.
</p>

<p align="center">
<img src="./img/Robustness test.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 4.</b> Robustness test results.
</p>

## Contact
If you have any questions, feel free to contact Polycarp  through Email (ypolycarp@gmail.com) or Github issues. Pull requests are highly welcomed!

## Acknowledgments
We thank the Computer Networks Lab, School of Electronics and Information Engineering, Liaoning Technical University for the computing infrastructure provided for our experiments.
We also thank [Informer](https://github.com/zhouhaoyi/Informer2020), [ETDataset](https://github.com/zhouhaoyi/ETDataset), [iTransformer](https://github.com/thuml/iTransformer) and [Autoformer](https://github.com/thuml/Autoformer) for the codebase of this training pipeline and dataset.
