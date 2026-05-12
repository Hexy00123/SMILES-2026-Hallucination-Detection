# Solution

## 1. Reproducibility Instructions
The repository is self-contained. To reproduce the results and generate the predictions.csv file, run the following commands:

```bash
git clone https://github.com/Hexy00123/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
```

Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run:
```bash
python solution.py
```

## 2. Final Solution Description
The final approach relies on extracting a high-dimensional set of raw hidden states targeted at the model's response, followed by a strongly regularized gradient boosting classifier to prevent overfitting (it was the main problem of this task as we almost initially had more features than samples).

### 2.1 Feature Extraction (aggregation.py)

- Layer Selection: I selected layers [12, 16, 22, 23]. Middle layers (12, 16) capture the core semantic meaning and factual representation, while the final layers (22, 23) encode the model's generation confidence. Early layers were discarded as they primarily represent basic syntax and add noise, last layer showed itself worse rather than pre-last and pre-pre-last.
- Dynamic Response-Biased Pooling: Since hallucination only occurs in the generated response (which is appended to the prompt), pooling the entire sequence dilutes the signal. Instead of hardcoding a fixed token window, I extracted the representation of the last single token and the mean pooling of the last 25% of tokens. This 25% window reliably isolates the generated response regardless of the prompt's length.
- Output: Concatenating these views across 4 layers results in a flat vector of 7168 features per sample.

### 2.2 Classifier (probe.py)

- The baseline MLP was replaced with a CatBoostClassifier.
- Strong Regularization: Feeding 7168 features from only 440 training samples is highly prone to overfitting. To counter this, I applied strict constraints: shallow trees (depth=3), high L2 regularization (l2_leaf_reg=30), and aggressive feature subsampling (colsample_bylevel=0.1).
- Sampling only 10% of features at each tree level forces the model to build diverse, decorrelated trees.

### 2.3 Validation Strategy (split.py)
- 5-fold StratifiedKFold. Within each fold, a 20% internal validation set is carved out from the training data using train_test_split.

## 3. Experiments and Failed Attempts
Several approaches were tested but they did not show as good test performance:

- Dimensionality Reduction via PCA: I attempted to compress the 7168 features down to 64 components using PCA to prevent overfitting. This failed (Test AUROC dropped to ~73%). PCA maximizes variance, which in LLM embeddings often correlates with general topic, effectively the hallucination signal.
- Blending Linear Models with Trees: I tried ensembling the CatBoost predictions with a regularized Logistic Regression. This approach degraded performance because linear models struggled to find generalized decision boundaries in the dense space, whereas CatBoost handled it naturally via column subsampling.
- Hand-crafted Geometric Features: I implemented custom geometric metrics, such as L2 norms (to measure confidence) and inter-layer cosine similarities (to measure representation drift). These were excluded from the final pipeline because they highly correlated with the raw embeddings and did not provide orthogonal information, serving only to introduce noise to the decision trees.

## 4. Final Results Summary
Below is the truncated evaluation summary produced by the final pipeline (averaged over 5 folds):
```
============================================================
 Hallucination Detection — Evaluation Summary
 (averaged over 5 folds)
============================================================
  Checkpoint                           Accuracy      F1   AUROC
------------------------------------------------------------
  1. Majority-class baseline             70.10%  82.42%     N/A
  2. Probe (train split)                 99.64%  99.74% 100.00%
  3. Probe (val split)                   78.56%  86.08%  79.59%
  4. Probe (test split)                  76.92%  84.87%  76.71%
------------------------------------------------------------
  Feature dim  : 7168
  Total samples: 689
  Folds        : 5
============================================================

★  Primary metric — Test AUROC: 76.71%
```

You can also find markuped predictions.csv here:
https://disk.yandex.ru/d/7YASBKcUheNGzg
