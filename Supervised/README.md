# 🤖 Supervised Machine Learning — Reference Guide

> A comprehensive cheat-sheet covering problem types, algorithm selection, loss functions, optimization techniques, and essential ML jargon.

---

## 📌 Table 1: Problem Types → Best-Fit Algorithms

| # | Problem Type | Description | Best Algorithm(s) |
|---|---|---|---|
| 1 | **Simple Linear Relationship** | Continuous target, linear relationship with features | Linear Regression, Ridge, Lasso |
| 2 | **Multicollinearity in features** | Features are highly correlated with each other | Ridge Regression (L2), PCA + Linear Regression |
| 3 | **Feature selection needed** | Many irrelevant features, need sparsity | Lasso Regression (L1), Elastic Net |
| 4 | **Both L1 + L2 regularization needed** | Need balance between feature selection & shrinkage | Elastic Net Regression |
| 5 | **Binary Classification** | Predict one of two classes (spam/not spam) | Logistic Regression, SVM, Random Forest, XGBoost |
| 6 | **Multi-class Classification** | Predict one of 3+ classes (cat/dog/bird) | Softmax Regression, Decision Tree, Random Forest, XGBoost, LightGBM |
| 7 | **Multi-label Classification** | Each sample can belong to multiple classes | Binary Relevance + any classifier, LightGBM |
| 8 | **Imbalanced Classes** | Target classes heavily skewed (fraud detection) | XGBoost (`scale_pos_weight`), LightGBM, SVM with `class_weight` |
| 9 | **High-dimensional sparse data** | Text/NLP, one-hot encoded with 1000s of features | Logistic Regression, Naive Bayes, Linear SVM |
| 10 | **Non-linear decision boundaries** | Classes not separable by a line/plane | Kernel SVM (RBF), Random Forest, Gradient Boosting, MLP |
| 11 | **Small dataset, well-structured** | Tabular, <10k rows | Decision Tree, KNN, SVM, Logistic Regression |
| 12 | **Large tabular dataset** | Millions of rows, mixed feature types | LightGBM, XGBoost, CatBoost |
| 13 | **Categorical features (high cardinality)** | Features like City, Product ID with many values | CatBoost, LightGBM (native cat support) |
| 14 | **Missing data present** | Dataset has many NaN/null values | XGBoost (handles natively), LightGBM, CatBoost |
| 15 | **Outliers in target variable** | Heavy-tailed regression targets | Huber Regression, Quantile Regression, Random Forest |
| 16 | **Probabilistic output needed** | Need confidence scores, not just predictions | Logistic Regression, Naive Bayes, Calibrated SVM |
| 17 | **Ranking / Ordinal output** | Output has an order (ratings 1–5) | Ordinal Regression, Gradient Boosting with custom loss |
| 18 | **Survival / Time-to-event** | Predict when an event will occur | Cox Regression, Survival Trees, XGBoost (AFT) |
| 19 | **Count data (non-negative integers)** | Predict # of events (customer visits) | Poisson Regression, Negative Binomial Regression |
| 20 | **Interpretability is critical** | Model needs to be explainable to stakeholders | Linear/Logistic Regression, Decision Tree, Rule-based models |
| 21 | **Speed / low latency at inference** | Model must predict in microseconds | Linear Regression, Logistic Regression, LightGBM |
| 22 | **Ensemble / Maximum accuracy** | Kaggle-style, accuracy is everything | XGBoost, LightGBM, CatBoost, Stacking, Blending |
| 23 | **Text classification** | Classify documents, sentiment, intent | Naive Bayes, Logistic Regression + TF-IDF, XGBoost |
| 24 | **Structured + Unstructured mix** | Tabular + text/image features combined | Gradient Boosting + embeddings, CatBoost |
| 25 | **Regression with feature interactions** | Non-linear interactions between features matter | Polynomial Regression, Gradient Boosting, Random Forest |
| 26 | **Online / incremental learning** | Model must update with each new data point | SGD Classifier/Regressor, Passive-Aggressive Classifier |
| 27 | **Robust to noisy labels** | Training data has mislabeled examples | Gradient Boosting (early stopping), SVM, Robust Regression |
| 28 | **Quantile / interval prediction** | Predict a range (10th–90th percentile) | Quantile Regression, Gradient Boosting with quantile loss |
| 29 | **Low training time constraint** | Fast to train as well as predict | LightGBM, Logistic Regression, Naive Bayes |
| 30 | **Domain: Medical / clinical** | Tabular, interpretability + accuracy needed | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| 31 | **Domain: Finance / credit scoring** | Scorecard, audit-ready | Logistic Regression (WoE), Decision Tree, XGBoost |
| 32 | **Domain: E-commerce (CTR prediction)** | Binary, sparse, huge scale | LightGBM, Logistic Regression, Factorization Machines |
| 33 | **Domain: Sensor / IoT time series** | Time-ordered tabular features with lag | Gradient Boosting on lag features, Random Forest |

---

## 📌 Table 2: Algorithm → Loss Function → Optimization Technique

| Algorithm | Task | Loss Function | Optimization Technique |
|---|---|---|---|
| **Linear Regression** | Regression | Mean Squared Error (MSE) | OLS (Closed-form) / Gradient Descent |
| **Ridge Regression** | Regression | MSE + L2 Penalty (λ‖w‖²) | Closed-form / Gradient Descent |
| **Lasso Regression** | Regression | MSE + L1 Penalty (λ‖w‖₁) | Coordinate Descent, Sub-gradient |
| **Elastic Net** | Regression | MSE + α·L1 + (1-α)·L2 | Coordinate Descent |
| **Polynomial Regression** | Regression | MSE | OLS on engineered features |
| **Huber Regression** | Regression | Huber Loss (quadratic near 0, linear in tails) | Gradient Descent / IRLS |
| **Quantile Regression** | Regression | Pinball / Quantile Loss | Linear Programming / SGD |
| **Poisson Regression** | Count Regression | Negative Log-Likelihood (Poisson) | IRLS / Newton-Raphson |
| **Logistic Regression** | Binary Classification | Binary Cross-Entropy (Log Loss) | Gradient Descent, L-BFGS, Newton-CG |
| **Softmax / Multinomial LR** | Multi-class | Categorical Cross-Entropy | Gradient Descent, L-BFGS |
| **Naive Bayes (Gaussian)** | Classification | Negative Log-Likelihood | Closed-form (MLE) |
| **Naive Bayes (Multinomial)** | Text Classification | Negative Log-Likelihood | Closed-form (MLE with Laplace smoothing) |
| **Decision Tree (CART)** | Both | MSE (regr.) / Gini / Entropy (clf.) | Greedy recursive splitting |
| **Decision Tree (ID3/C4.5)** | Classification | Information Gain / Gain Ratio | Greedy splitting |
| **Random Forest** | Both | MSE / Gini / Entropy (avg. over trees) | Bagging + random feature subsets |
| **Extra Trees** | Both | MSE / Gini / Entropy | Bagging + random splits (more randomness) |
| **AdaBoost** | Classification | Exponential Loss | Forward Stagewise Additive Modeling |
| **Gradient Boosting (sklearn)** | Both | MSE / MAE / Log Loss / Deviance | Gradient Descent in function space |
| **XGBoost** | Both | Custom (default: MSE/Log Loss) + L1+L2 reg | Newton Boosting (2nd-order Taylor expansion) |
| **LightGBM** | Both | Custom (default: MSE/Log Loss) | Gradient-based One-Side Sampling (GOSS) + EFB |
| **CatBoost** | Both | Custom (default: Logloss/RMSE) | Ordered Boosting + Newton step |
| **Support Vector Machine (SVM)** | Classification | Hinge Loss | SMO (Sequential Minimal Optimization) |
| **SVM — Regression (SVR)** | Regression | ε-insensitive Loss | SMO / Quadratic Programming |
| **KNN Classifier** | Classification | N/A (lazy learner) | None (memorize training data) |
| **KNN Regressor** | Regression | N/A (lazy learner) | None |
| **Linear Discriminant Analysis** | Classification | N/A (generative model) | Eigendecomposition (closed-form) |
| **Quadratic Discriminant Analysis** | Classification | N/A (generative model) | Closed-form (class-wise covariance) |
| **Perceptron** | Classification | Perceptron Loss (0-1 approx.) | Online Perceptron Update Rule |
| **SGD Classifier/Regressor** | Both | Configurable (Hinge/Log/MSE) | Stochastic Gradient Descent |
| **Passive-Aggressive Classifier** | Online Clf. | Hinge Loss (modified) | Passive-Aggressive Update |
| **Bayesian Ridge Regression** | Regression | Gaussian Log-Likelihood | Empirical Bayes (EM / closed-form) |

---

## 📌 Table 3: ML Jargon Glossary

| Term | What it actually means |
|---|---|
| **Bias** | Error from wrong assumptions in the model — underfitting. A biased model is too simple to capture the truth. |
| **Variance** | Error from sensitivity to small fluctuations in training data — overfitting. High-variance model memorizes noise. |
| **Bias-Variance Tradeoff** | The tension between a model being too simple (high bias) vs. too complex (high variance). You tune this via regularization, depth, etc. |
| **Overfitting** | Model performs great on train data, terribly on unseen data. It learned the noise, not the signal. |
| **Underfitting** | Model is too simple to learn even the training data well. Low train AND test accuracy. |
| **Regularization** | Adding a penalty for model complexity to prevent overfitting. L1 (Lasso) shrinks weights to zero; L2 (Ridge) shrinks them smoothly. |
| **Feature Engineering** | Creating new features from raw data — ratios, log transforms, lag features, interaction terms — to help the model learn better. |
| **Feature Importance** | A score indicating how much each feature contributed to the model's predictions. Common in tree-based models. |
| **Cross-Validation** | Technique to evaluate model on multiple train/val splits (e.g., k-fold) to get a robust performance estimate. |
| **Hyperparameter Tuning** | Finding the best settings for your model (learning rate, max depth, etc.) using Grid Search, Random Search, or Bayesian Optimization. |
| **Loss Function** | The mathematical function the model minimizes during training. It measures prediction error. |
| **Objective Function** | Loss function + regularization terms. What the optimizer actually minimizes. |
| **Gradient Descent** | Iterative optimization — compute gradient of loss w.r.t. parameters, step in the negative gradient direction. |
| **Learning Rate** | How large each gradient descent step is. Too large = diverge; too small = slow convergence. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Batch Size** | Number of samples per gradient update step (Mini-batch GD). |
| **Convergence** | When the model's loss stops significantly improving — training is "done". |
| **Early Stopping** | Stop training when validation loss stops improving, to prevent overfitting. |
| **Bootstrapping** | Sampling training data *with replacement*. Foundation of Bagging / Random Forest. |
| **Bagging** | Train many models on bootstrap samples, aggregate predictions. Reduces variance. |
| **Boosting** | Train models sequentially where each corrects the errors of the previous one. Reduces both bias and variance. |
| **Stacking** | Train a meta-model on the outputs of multiple base models. Often wins in competitions. |
| **Blending** | Like stacking but simpler — average or weighted average of predictions from different models. |
| **Precision** | Of all samples predicted positive, how many actually are positive? TP / (TP + FP). |
| **Recall (Sensitivity)** | Of all actual positives, how many did we catch? TP / (TP + FN). |
| **F1 Score** | Harmonic mean of Precision and Recall. Good metric when classes are imbalanced. |
| **AUC-ROC** | Area under the ROC curve. Measures model's ability to discriminate between classes across all thresholds. 0.5 = random, 1.0 = perfect. |
| **Log Loss** | Measures probability predictions. Penalizes confident wrong predictions heavily. Lower is better. |
| **RMSE** | Root Mean Squared Error. Common regression metric. Sensitive to outliers. |
| **MAE** | Mean Absolute Error. Robust regression metric. Less sensitive to outliers than RMSE. |
| **R² Score** | Proportion of variance in target explained by the model. 1.0 = perfect, 0.0 = predicts mean. |
| **Multicollinearity** | When two or more features are highly correlated. Hurts linear models' coefficient interpretability. |
| **Heteroscedasticity** | When the variance of errors is not constant across all predictions. Violates linear regression assumptions. |
| **Imputation** | Filling in missing values — with mean, median, mode, or model-based predictions. |
| **One-Hot Encoding** | Converting a categorical variable into binary columns — one per category. |
| **Label Encoding** | Assigning an integer to each category. Fast but implies an order that may not exist. |
| **Target Encoding** | Replace a category with the mean target value for that category. Powerful but risks data leakage. |
| **Data Leakage** | When information from the future or from the test set leaks into training. Causes falsely optimistic metrics. |
| **Train/Val/Test Split** | Dividing data: train (model learns), val (tune hyperparams), test (final unbiased evaluation). |
| **SHAP Values** | SHapley Additive exPlanations — a game-theory-based approach to explain individual predictions from any model. |
| **Calibration** | Ensuring predicted probabilities reflect true frequencies (e.g., 0.7 predicted probability should be correct 70% of the time). |
| **Class Imbalance** | When one class vastly outnumbers another. Requires techniques like SMOTE, class_weight, or threshold tuning. |
| **SMOTE** | Synthetic Minority Over-sampling Technique — generates synthetic samples for the minority class. |
| **Threshold Tuning** | Changing the classification decision boundary (default 0.5) to balance Precision vs Recall for a use case. |
| **Inductive Bias** | The set of assumptions a model makes about the data. E.g., Linear models assume linearity. |
| **No Free Lunch Theorem** | No single algorithm is best for all problems. Performance depends entirely on the data distribution. |
| **Curse of Dimensionality** | As features increase, data becomes sparse and distance-based methods fail. Needs dimensionality reduction. |
| **Occam's Razor (in ML)** | Prefer simpler models unless a complex one clearly wins. Complexity should be justified by evidence. |
| **p-value** | Probability of observing results at least as extreme as the data, assuming the null hypothesis is true. p < 0.05 is commonly used as significance. |
| **Confidence Interval** | A range of values within which the true parameter lies with a given probability (e.g., 95%). |
| **OOB Score** | Out-of-Bag score — Random Forest's built-in validation using samples not chosen in each bootstrap. |
| **Leaf Node** | Terminal node in a decision tree. Makes the final prediction. |
| **Pruning** | Cutting branches from a decision tree post-training to reduce overfitting. |
| **Gini Impurity** | Measure of how often a randomly chosen element would be incorrectly classified. Used in CART trees. |
| **Information Gain (Entropy)** | Reduction in entropy after a dataset split. Used in ID3/C4.5 trees. |
| **Shrinkage** | In boosting, each tree's contribution is scaled by the learning rate. Prevents overfitting. |
| **Column Subsampling** | Randomly selecting a fraction of features per tree/split. Used in XGBoost, LightGBM, RF. |
| **Histogram-based Splitting** | LightGBM's trick: bin continuous features into histograms for ultra-fast split finding. |
| **Leaf-wise Growth** | LightGBM grows trees leaf-by-leaf (deepest leaf first) vs. level-by-level. Faster, but needs `num_leaves` tuning. |
| **Newton Boosting** | XGBoost's trick: use 2nd-order Taylor expansion (gradient + Hessian) for faster, more accurate updates. |
| **Softmax** | Function that converts raw scores (logits) into probabilities summing to 1. Used for multi-class output. |
| **Logit** | The raw output score before applying sigmoid or softmax. Also called log-odds. |
| **Sigmoid** | S-shaped function squashing any value to (0, 1). Used in binary classification for probability output. |
| **Kernel Trick** | Implicitly mapping data to higher dimensions via a kernel function (e.g., RBF) without computing the transformation. Used in SVM. |
| **Support Vectors** | The training samples closest to the decision boundary in SVM. They define the margin. |
| **Margin** | The distance between the decision boundary and the nearest support vectors. SVM maximizes this. |

---

*Generated for ML practitioners — from Linear Regression to LightGBM.*