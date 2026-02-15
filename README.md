# ML Assignment 2 — Classification Models with Streamlit Deployment

## a. Problem Statement

The objective of this project is to build, evaluate, and deploy **six machine learning classification models** to predict whether an online shopping session will result in a purchase (revenue generation). The target variable `Revenue` is binary (`True` = purchase made, `False` = no purchase), making this a **binary classification** problem.

The models are trained on the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository. The goal is to compare the performance of different classification algorithms — including traditional models (Logistic Regression, Decision Tree, kNN, Naive Bayes) and ensemble methods (Random Forest, XGBoost) — using multiple evaluation metrics, and to deploy an interactive Streamlit web application for demonstration.

---

## b. Dataset Description

| Property | Details |
|---|---|
| **Dataset Name** | Online Shoppers Purchasing Intention Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online-shoppers-purchasing-intention-dataset) |
| **Number of Instances** | 12,330 |
| **Number of Features** | 17 (input features) |
| **Target Variable** | `Revenue` (Binary: True / False) |
| **Task** | Binary Classification |
| **Class Distribution** | False: 10,422 (84.53%) · True: 1,908 (15.47%) — imbalanced dataset |

### Feature Details

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | Administrative | Numeric | Number of administrative pages visited |
| 2 | Administrative_Duration | Numeric | Total time spent on administrative pages (seconds) |
| 3 | Informational | Numeric | Number of informational pages visited |
| 4 | Informational_Duration | Numeric | Total time spent on informational pages (seconds) |
| 5 | ProductRelated | Numeric | Number of product-related pages visited |
| 6 | ProductRelated_Duration | Numeric | Total time spent on product-related pages (seconds) |
| 7 | BounceRates | Numeric | Average bounce rate of pages visited |
| 8 | ExitRates | Numeric | Average exit rate of pages visited |
| 9 | PageValues | Numeric | Average page value of pages visited |
| 10 | SpecialDay | Numeric | Closeness of the visit to a special day (e.g., Valentine's Day) |
| 11 | Month | Categorical | Month of the year |
| 12 | OperatingSystems | Numeric | Operating system of the visitor |
| 13 | Browser | Numeric | Browser used by the visitor |
| 14 | Region | Numeric | Geographic region of the visitor |
| 15 | TrafficType | Numeric | Traffic source type |
| 16 | VisitorType | Categorical | Returning visitor, new visitor, or other |
| 17 | Weekend | Boolean | Whether the visit occurred on a weekend |

### Preprocessing Steps
- Dropped rows with missing values
- Encoded categorical features (`Month`, `VisitorType`) using Label Encoding
- Converted boolean features (`Weekend`, `Revenue`) to integer (0/1)
- Applied `StandardScaler` for feature normalization
- Train-test split: 80% training (9,864 samples) / 20% testing (2,466 samples), stratified

---

## c. Models Used

Six classification models were implemented and evaluated on the same dataset:

1. **Logistic Regression** — Linear model with L2 regularization (solver: lbfgs, max_iter: 1000)
2. **Decision Tree Classifier** — Tree-based model (max_depth: 10, min_samples_split: 5, min_samples_leaf: 2)
3. **K-Nearest Neighbor (kNN) Classifier** — Distance-based model (k=7, weights: distance, metric: minkowski)
4. **Gaussian Naive Bayes** — Probabilistic model based on Bayes' theorem with Gaussian likelihood
5. **Random Forest (Ensemble)** — Bagging ensemble of 200 decision trees (max_depth: 15)
6. **XGBoost (Ensemble)** — Gradient boosting ensemble of 200 trees (learning_rate: 0.1, max_depth: 6)

### Model Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8832 | 0.8652 | 0.7640 | 0.3560 | 0.4857 | 0.4696 |
| Decision Tree | 0.8800 | 0.8235 | 0.6295 | 0.5471 | 0.5854 | 0.5174 |
| kNN | 0.8771 | 0.8004 | 0.6927 | 0.3717 | 0.4838 | 0.4475 |
| Naive Bayes | 0.7794 | 0.8020 | 0.3802 | 0.6728 | 0.4858 | 0.3826 |
| Random Forest (Ensemble) | 0.9015 | 0.9209 | 0.7340 | 0.5707 | 0.6421 | 0.5923 |
| XGBoost (Ensemble) | 0.8954 | 0.9237 | 0.6987 | 0.5707 | 0.6282 | 0.5720 |

### Best Model per Metric

| Metric | Best Model | Score |
|---|---|---|
| Accuracy | Random Forest (Ensemble) | 0.9015 |
| AUC | XGBoost (Ensemble) | 0.9237 |
| Precision | Logistic Regression | 0.7640 |
| Recall | Naive Bayes | 0.6728 |
| F1 Score | Random Forest (Ensemble) | 0.6421 |
| MCC | Random Forest (Ensemble) | 0.5923 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieves a strong accuracy of 88.32% and the **highest precision (0.7640)** among all models, meaning when it predicts a purchase, it is correct ~76% of the time. However, it suffers from very **low recall (0.3560)**, indicating that it misses approximately 64% of actual purchasers. This is characteristic of linear models on imbalanced datasets — they tend to favour the majority class. The AUC of 0.8652 is reasonable, suggesting the model has decent discriminatory ability, but the F1 score (0.4857) and MCC (0.4696) reveal that the overall positive-class performance is limited. Logistic Regression is best suited when minimizing false positives is critical (e.g., targeted marketing campaigns with limited budget). |
| **Decision Tree** | Decision Tree delivers a balanced trade-off between precision (0.6295) and recall (0.5471), resulting in the **second-highest F1 score (0.5854)** among individual (non-ensemble) models. Its recall is significantly better than Logistic Regression and kNN, capturing ~55% of actual purchasers. The AUC of 0.8235 is the lowest among tree-based models, indicating moderate ranking ability. Despite hyperparameter tuning (max_depth=10), Decision Trees are prone to overfitting on training data, and the lower precision compared to Logistic Regression shows it generates more false positives. It is interpretable and useful for understanding feature importance, but ensemble methods clearly outperform it. |
| **kNN** | K-Nearest Neighbor achieves the **lowest accuracy (0.8771)** among non-Naive-Bayes models and a low AUC (0.8004). With a recall of only 0.3717, it identifies fewer than 38% of actual purchasers. The precision of 0.6927 is moderate. The poor recall and F1 (0.4838) suggest that kNN struggles with this dataset, likely due to the curse of dimensionality — with 17 features, distance-based approaches become less effective as the feature space is high-dimensional. Additionally, the imbalanced class distribution further degrades kNN's ability to correctly classify the minority class. kNN is computationally expensive at inference time and is not well-suited for this type of dataset. |
| **Naive Bayes** | Gaussian Naive Bayes has the **lowest accuracy (0.7794)** overall but achieves the **highest recall (0.6728)**, correctly identifying ~67% of actual purchasers — significantly more than any other model. However, this comes at the cost of very **low precision (0.3802)**, meaning ~62% of its positive predictions are false alarms. The AUC (0.8020) is comparable to kNN. The low MCC (0.3826) indicates weak overall correlation between predictions and ground truth. Naive Bayes works well when the independence assumption holds, but the features in this dataset are correlated (e.g., page visit counts and durations), which violates this assumption. It is best suited for scenarios where catching all potential buyers is more important than precision — such as broad promotional campaigns. |
| **Random Forest (Ensemble)** | Random Forest is the **best overall performer**, achieving the **highest accuracy (0.9015)**, **highest F1 score (0.6421)**, and **highest MCC (0.5923)**. It maintains a strong balance between precision (0.7340) and recall (0.5707), meaning it captures ~57% of actual purchasers while keeping false positives low. The AUC of 0.9209 demonstrates excellent discriminatory ability. As a bagging ensemble of 200 decision trees, Random Forest mitigates the overfitting tendency of individual decision trees through bootstrap aggregation and random feature selection. It is robust to noise, handles non-linear relationships well, and provides reliable feature importance rankings. This model is the recommended choice for deployment in this classification task. |
| **XGBoost (Ensemble)** | XGBoost achieves the **highest AUC (0.9237)**, indicating the best overall ranking and probability calibration ability. Its accuracy (0.8954) and recall (0.5707) are close to Random Forest, with slightly lower precision (0.6987) and F1 score (0.6282). The MCC (0.5720) is the second-highest. XGBoost uses sequential gradient boosting, where each tree corrects the errors of the previous ones, making it highly effective for structured/tabular data. The close performance to Random Forest validates that ensemble methods significantly outperform individual classifiers on this dataset. XGBoost is particularly useful when probability estimates and ranking (AUC) are important, such as in lead scoring or customer conversion probability estimation. |

---

## Project Structure

```
ml-classification-app/
│── app.py                          # Streamlit web application
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation
│── model/
│   ├── train_models.py             # Model training and evaluation script
│   ├── logistic_regression.pkl     # Saved Logistic Regression model
│   ├── decision_tree.pkl           # Saved Decision Tree model
│   ├── knn.pkl                     # Saved kNN model
│   ├── naive_bayes.pkl             # Saved Naive Bayes model
│   ├── random_forest_ensemble.pkl  # Saved Random Forest model
│   ├── xgboost_ensemble.pkl        # Saved XGBoost model
│   ├── scaler.pkl                  # Saved StandardScaler
│   ├── label_encoders.pkl          # Saved Label Encoders
│   ├── feature_names.json          # Feature names list
│   ├── model_metrics.csv           # Evaluation metrics for all models
│   ├── confusion_matrices.pkl      # Confusion matrices
│   ├── classification_reports.pkl  # Classification reports
│   └── test_data.csv              # Test data for app demo
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sravanibits/bits-ml-assignment2.git
cd bits-ml-assignment2
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Train the models (generates saved model files)
```bash
python model/train_models.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## Streamlit App Features

- **📁 Dataset Upload (CSV):** Upload your own test CSV data for prediction
- **🔧 Model Selection Dropdown:** Choose from any of the 6 trained models
- **📈 Evaluation Metrics Display:** View Accuracy, AUC, Precision, Recall, F1, and MCC
- **📊 Confusion Matrix:** Visual heatmap of prediction results
- **📋 Classification Report:** Detailed per-class precision, recall, and F1 scores
- **📉 Visual Comparisons:** Bar charts and radar charts comparing all models

---

## Technologies Used

- **Python 3.11**
- **Scikit-learn** — Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
- **XGBoost** — Gradient Boosted Trees
- **Streamlit** — Interactive web application framework
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Visualization
- **Streamlit Community Cloud** — Deployment

---

## Links

- **GitHub Repository:** [[https://github.com/sravanibits/bits-ml-assignment2]([https://github.com/sravanigbits/ml-assignment-2/tree/main)](https://github.com/sravanibits/bits-ml-assignment2](https://github.com/sravanigbits/ml-assignment-2/tree/main))
- **Live Streamlit App:** [https://bits-ml-assignment-2-sravani.streamlit.app/](https://bits-ml-assignment-2-sravani.streamlit.app/)

---

*ML Assignment 2 | M.Tech AIML/DSE | BITS Pilani — WILP*
