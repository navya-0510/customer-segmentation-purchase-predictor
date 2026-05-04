# Customer Segmentation and Purchase Predictor

A complete AI/ML pipeline built **entirely from scratch in pure Python** — no NumPy, no Pandas, no Scikit-learn. Every algorithm, every formula, and every metric is implemented using only Python lists, loops, and the built-in `math` module.

Built as part of the **Artificial Intelligence and Machine Learning Lab**  
Jaypee University of Information Technology, Waknaghat  
Under the guidance of **Dr. Meghna Dhalaria**, Assistant Professor, Dept. of CSE/IT

---

## What this project does

- **Segments customers** into groups using K-Means Clustering (unsupervised)
- **Predicts purchase behaviour** (will a customer buy or not?) using KNN, Decision Tree, and ANN (supervised)
- **Evaluates** all models using Accuracy, Precision, Recall, and F1-Score — computed from scratch

---

## Algorithms implemented

| Algorithm | Type | Purpose |
|---|---|---|
| K-Means Clustering | Unsupervised | Group customers into segments |
| K-Nearest Neighbors (KNN) | Supervised | Predict purchase label |
| Decision Tree | Supervised | Predict purchase label |
| Artificial Neural Network (ANN) | Supervised | Predict purchase label |

---

## Project structure

```
customer-segmentation-purchase-predictor/
│
├── Customer_Segmentation_and_Purchase_Predictor.ipynb   ← main notebook
└── README.md
```

### Notebook cells

| Cell | Content |
|---|---|
| Cell 1 | Data generation — 120 synthetic customers |
| Cell 2 | Preprocessing — min-max normalization, train/test split, Euclidean distance |
| Cell 3 | K-Means Clustering (K=3) |
| Cell 4 | KNN Classifier (K=5) |
| Cell 5 | Decision Tree (max depth=5, Gini impurity) |
| Cell 6 | ANN (3→4→1 architecture, sigmoid, backpropagation, 500 epochs) |
| Cell 7 | Evaluation metrics — Accuracy, Precision, Recall, F1, Confusion Matrix |
| Cell 8 | Visualisations — cluster scatter plot, ANN loss curve, model comparison bar chart |

---

## Dataset

Synthetic dataset of **120 customers** generated using Python's `random` module.

| Feature | Range | Description |
|---|---|---|
| `age` | 18–65 | Customer age |
| `annual_income` | ₹20,000–₹1,20,000 | Yearly income |
| `spending_score` | 1–100 | Spending tendency score |
| `will_purchase` | 0 or 1 | Label: 0 = won't buy, 1 = will buy |

- **Train set:** 96 samples (80%)
- **Test set:** 24 samples (20%)
- **Random seed:** 42 (fully reproducible)

---

## How to run

### On Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `Customer_Segmentation_and_Purchase_Predictor.ipynb`
4. Click **Runtime → Run all**

### Locally

```bash
pip install matplotlib
jupyter notebook Customer_Segmentation_and_Purchase_Predictor.ipynb
```

Run cells top to bottom in order — each cell depends on variables from the previous one.

---

## Key concepts implemented from scratch

**K-Means Clustering**
- Random centroid initialisation
- Euclidean distance assignment
- Centroid recomputation (mean of cluster)
- Convergence check

**KNN (K-Nearest Neighbors)**
- Euclidean distance to all training points
- Sort and select K=5 nearest neighbors
- Majority vote prediction

**Decision Tree**
- Gini impurity: `1 − p₀² − p₁²`
- Weighted Gini for split selection
- Recursive tree building (max depth = 5)
- Tree stored as nested Python dictionaries

**Artificial Neural Network (ANN)**
- Architecture: 3 inputs → 4 hidden neurons → 1 output
- Activation: Sigmoid `σ(x) = 1 / (1 + e⁻ˣ)`
- Loss: Mean Squared Error (MSE)
- Training: Backpropagation + Gradient Descent (lr = 0.1, epochs = 500)
- Weights initialised as random values in [−0.5, +0.5]
- Online gradient descent — weights updated after every sample

**Evaluation Metrics**
- Confusion matrix (TP, TN, FP, FN)
- Accuracy = (TP + TN) / total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 × P × R / (P + R)

---

## Visualisations produced

**K-Means Cluster Scatter Plot** — customers plotted by income vs spending score, coloured by cluster (red, blue, green), with centroids marked as black stars.

**ANN Training Loss Curve** — MSE loss plotted over 500 epochs, showing the network converging as it learns.

**Model Comparison Bar Chart** — grouped bar chart comparing KNN, Decision Tree, and ANN across all four metrics side by side.

---

## Libraries used

| Library | Version | Purpose |
|---|---|---|
| `random` | built-in | Data generation, weight initialisation |
| `math` | built-in | `sqrt` for distance, `exp` for sigmoid |
| `matplotlib` | any | Visualisation only |

No NumPy. No Pandas. No Scikit-learn. No TensorFlow.

---

## Limitations

- Dataset is synthetic — results on real customer data may differ
- ANN uses online gradient descent — mini-batch would be more stable
- Decision Tree training is O(n × f × t) — slow on large datasets
- KNN stores all training data — memory-intensive at scale
- Single 80/20 split — no cross-validation
- MSE used as ANN loss — Binary Cross-Entropy is more appropriate for classification

---

## Author

**Navya Navya** 
