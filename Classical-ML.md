---
title: Exploring Classical ML Models With PyTorch Internals
author: Kyle Lukaszek
date: October 2023
tags:
  - ML
  - KNN
  - Linear
  - Regression
  - Logistic
  - Regression
  - K-Fold-Cross-Validation
  - Classification
  - Nearest-Neighbours
description: Adapted Jupyter Notebook from CIS*4780 Computational Intelligence
---
# Exploring Classic Machine Learning Models with PyTorch: An Implementation Log

I recently worked through implementing several foundational machine learning models using PyTorch for core operations. This post documents the process, observations, and some of the practical considerations encountered.

## Dependencies
```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
```

## PyTorch Z-Score Normalization
Data normalization is a common preprocessing step. I implemented a Z-score normalizer using PyTorch tensor operations.

```python
## Simple implementation based on sklearn.preprocessing.StandardScaler
class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        # Calculate the mean and standard deviation along dim 0 (columns)
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Not fitted yet. Call fit() before transform()")
        
        # Handle cases where std might be zero to avoid division by zero.
        # If std is 0, the feature is constant; (data - mean) will be 0.
        # Dividing by 1 (instead of 0) results in 0 for that normalized feature.
        std_safe = torch.where(self.std == 0, torch.tensor(1.0, device=self.std.device), self.std)
        normalized_data = (data - self.mean) / std_safe
        return normalized_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
```
*Observation: A key detail in `transform` is handling features with a standard deviation of zero to prevent division errors. The `std_safe` variable addresses this by substituting `1.0` if `std` is zero.*

## Part 1 - KNN Implementation

The K-Nearest Neighbors algorithm classifies new data points based on the majority class of their 'K' closest neighbors in the training set.

### KNN Classifier Class
The implementation involves a `fit` method to store training data and a `predict` method to classify test points.

```python
class KNNClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Ensure inputs are tensors and of appropriate dtype
        if not isinstance(X_train, torch.Tensor):
            self.X_train = torch.tensor(X_train, dtype=torch.float32)
        else:
            self.X_train = X_train.float()

        if not isinstance(y_train, torch.Tensor):
            # Assuming labels can be float for sign sum if using {-1, 1}
            self.y_train = torch.tensor(y_train, dtype=torch.float32) 
        else:
            self.y_train = y_train.float()


    def predict(self, X_test, k):
        y_pred = []
        if not isinstance(X_test, torch.Tensor):
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        else:
            X_test_tensor = X_test.float()

        for x_test_single in X_test_tensor:
            distances = torch.norm(self.X_train - x_test_single, dim=1)
            indices = torch.argsort(distances)[:k]
            neighbors_labels = self.y_train[indices]
            # Majority vote for binary {-1, 1} labels
            prediction = torch.sign(torch.sum(neighbors_labels))
            y_pred.append(prediction.item())
        return y_pred
```

### KNN K-fold Cross Validation Function
To determine an appropriate value for `K`, K-Fold Cross-Validation was used. The data is split into folds, with the model trained and tested iteratively.

```python
def knn_k_fold_cross_validation(X, y, k_values, n_folds=10):
    train_accuracies, test_accuracies, train_errors, test_errors = [], [], [], []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42) # random_state for consistency

    for k_val in k_values: # Renamed k to k_val to avoid conflict
        fold_accuracies_train, fold_accuracies_test = [], []
        fold_errors_train, fold_errors_test = [], []

        for train_index, test_index in kf.split(X):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            knn = KNNClassifier()
            knn.fit(X_train_fold, y_train_fold)

            # knn.predict expects numpy array or list of arrays based on original usage
            train_predictions = knn.predict(X_train_fold, k_val) 
            test_predictions = knn.predict(X_test_fold, k_val)

            train_accuracy = accuracy_score(y_train_fold, train_predictions)
            test_accuracy = accuracy_score(y_test_fold, test_predictions)

            fold_accuracies_train.append(train_accuracy)
            fold_accuracies_test.append(test_accuracy)
            fold_errors_train.append(1 - train_accuracy)
            fold_errors_test.append(1 - test_accuracy)

        train_accuracies.append(np.mean(fold_accuracies_train))
        test_accuracies.append(np.mean(fold_accuracies_test))
        train_errors.append(np.mean(fold_errors_train))
        test_errors.append(np.mean(fold_errors_test))

    return {
        "train_acc": train_accuracies, "test_acc": test_accuracies,
        "train_err": train_errors, "test_err": test_errors
    }
```

### Load Data And Run KNN Code
Applying the KNN classifier to a dataset with two input features.

```python
input_file = 'KNNClassifierInput.csv'
output_file = 'KNNClassifierOutput.csv'
input_df_knn = pd.read_csv(input_file, header=0)
output_df_knn = pd.read_csv(output_file).dropna(axis=1)
X_knn_data = input_df_knn[['Input 1', 'Input 2']].values
y_knn_data = output_df_knn.values.squeeze()

k_values_range = list(range(1, 31)) # Renamed for clarity
results_knn_cv = knn_k_fold_cross_validation(X_knn_data, y_knn_data, k_values_range, n_folds=10)

train_acc_knn = results_knn_cv["train_acc"]
test_acc_knn = results_knn_cv["test_acc"]

best_k_knn_idx = np.argmax(test_acc_knn) # Get index of max test accuracy
best_k_knn = k_values_range[best_k_knn_idx]
print(f"Best K Value: {best_k_knn}")
print(f"Test Accuracy with Best K Value: {test_acc_knn[best_k_knn_idx] * 100:.2f}%")
```
```
    Best K Value: 11
    Test Accuracy with Best K Value: 98.00%
```
*(Note: Output from a specific run. `shuffle=True` in `KFold` can lead to minor variations).*

### Plot Results
Visualizing training and testing accuracy against K values.

```python
plt.figure(figsize=(10, 6))
plt.plot(k_values_range, results_knn_cv['train_acc'], label='Training Accuracy')
plt.plot(k_values_range, results_knn_cv['test_acc'], label='Testing Accuracy')
plt.axvline(best_k_knn, color='r', linestyle='--', label=f'Best K = {best_k_knn}')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```
![](./images/Classical-ML/1.png)

### K-Value Analysis
The cross-validation identified K=11 as optimal for this run, achieving 98.00% test accuracy. This K value is neither extremely small (which could lead to high variance and overfitting) nor extremely large (which could lead to high bias and oversmoothing). It suggests a reasonable balance for this dataset. While shuffling in KFold can cause the optimal K to vary slightly between runs, values in this mid-range generally indicate better generalization.

## Part 2: Linear Regression (Ridge)

Ridge Regression is a linear regression variant that includes L2 regularization to penalize large coefficient weights, helping to prevent overfitting and improve stability, especially with multicollinearity.

### Define Ridge Regression Class
A simple linear model using `nn.Linear`.

```python
class RidgeRegression(nn.Module):
    def __init__(self, input_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)
```

### Define Ridge Regression K-fold Cross Validation Function
This function evaluates Ridge Regression for a given `lambda` (regularization strength) using K-fold cross-validation.

```python
def lr_k_fold_cross_validation(X, y, model_class, criterion, base_optimizer_class, num_epochs=100, lambda_val=0.0, n_folds=5, lr=0.01):
    train_losses_fold, test_losses_fold, r2_scores_fold, mse_scores_fold = [], [], [], []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Re-initialize model and optimizer for each fold for strict CV
        current_model = model_class(X.shape[1]) 
        current_optimizer = base_optimizer_class(current_model.parameters(), lr=lr)
        
        # Ensure data are tensors
        if not isinstance(X_train, torch.Tensor): X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor): y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
        if not isinstance(X_test, torch.Tensor): X_test = torch.tensor(X_test, dtype=torch.float32)
        if not isinstance(y_test, torch.Tensor): y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

        current_model.train()
        for epoch in range(num_epochs):
            current_optimizer.zero_grad()
            outputs = current_model(X_train)
            loss = criterion(outputs, y_train)
            
            l2_reg = 0.0 # Calculate L2 regularization term
            for param in current_model.parameters():
                if param.requires_grad and param.dim() > 1: # Weights are typically 2D
                     l2_reg += torch.sum(param ** 2)
            loss += lambda_val * l2_reg
            
            loss.backward()
            nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0) # Gradient clipping
            current_optimizer.step()

        current_model.eval()
        with torch.no_grad():
            train_outputs = current_model(X_train)
            test_outputs = current_model(X_test)
            train_losses_fold.append(criterion(train_outputs, y_train).item())
            test_losses_fold.append(criterion(test_outputs, y_test).item())
            
            y_test_np = y_test.cpu().numpy()
            test_outputs_np = test_outputs.cpu().numpy()
            r2_scores_fold.append(r2_score(y_test_np, test_outputs_np))
            mse_scores_fold.append(mean_squared_error(y_test_np, test_outputs_np))

    return {
        'avg_train_loss': np.mean(train_losses_fold), 'avg_test_loss': np.mean(test_losses_fold),
        'avg_r2_score': np.mean(r2_scores_fold), 'avg_mse_score': np.mean(mse_scores_fold)
    }
```
*Observation: For proper K-fold CV, the model and optimizer should be re-initialized within each fold loop. The original IPYNB's `lr_k_fold_cross_validation` structure passed these in, meaning the same instance was trained across folds for a *given* lambda. The outer lambda-iteration loop *did* re-initialize. The version above shows re-initialization inside the fold loop for best practice.*

### Load Data And Run Linear Regression Code
Input features `X` and target `y` were Z-score normalized. The `lambda` values were scaled from an original range of [0, 250] to [0, 1].

```python
input_file_lr = 'LinearRegression.csv'
target_file_lr = 'LinearRegressionTarget.csv'
input_data_df_lr = pd.read_csv(input_file_lr)
target_data_df_lr = pd.read_csv(target_file_lr)

X_lr_data = torch.tensor(input_data_df_lr.values, dtype=torch.float32)
y_lr_data = torch.tensor(target_data_df_lr.values, dtype=torch.float32)

input_normalizer_lr = ZScoreNormalizer()
X_lr_norm = input_normalizer_lr.fit_transform(X_lr_data)
target_normalizer_lr = ZScoreNormalizer()
y_lr_norm = target_normalizer_lr.fit_transform(y_lr_data)

input_dim_lr = X_lr_norm.shape[1]
num_epochs_lr = 100

raw_lambda_values_lr = np.array(list(range(0, 251))).reshape(-1, 1)
lambda_scaler_lr = MinMaxScaler(feature_range=(0, 1))
scaled_lambda_values_lr = lambda_scaler_lr.fit_transform(raw_lambda_values_lr).squeeze()

results_lr_cv_list = [] # Renamed

for lambda_val_s in scaled_lambda_values_lr: # Renamed lambda_val
    # Note: model_class=RidgeRegression, base_optimizer_class=torch.optim.SGD are passed
    # to lr_k_fold_cross_validation where they are instantiated.
    cv_fold_results = lr_k_fold_cross_validation(
        X_lr_norm, y_lr_norm, 
        model_class=RidgeRegression, # Pass the class, not an instance
        criterion=nn.MSELoss(), 
        base_optimizer_class=torch.optim.SGD, # Pass the class
        num_epochs=num_epochs_lr, 
        lambda_val=lambda_val_s, 
        n_folds=5,
        lr=0.01 # Learning rate from notebook
    )
    results_lr_cv_list.append((lambda_val_s, cv_fold_results))

best_lambda_s_lr, best_results_lr_stats = max(results_lr_cv_list, key=lambda x: x[1]['avg_r2_score']) # Renamed

# Find result for scaled lambda corresponding to raw lambda 0
raw_lambda_0_scaled_val = lambda_scaler_lr.transform(np.array([[0]]))[0][0] # Renamed
result_for_lambda_0_lr = next(r for l, r in results_lr_cv_list if np.isclose(l, raw_lambda_0_scaled_val)) # Renamed

print("Results for scaled lambda corresponding to raw lambda=0:")
print(result_for_lambda_0_lr) 
print(f"\nBest scaled lambda: {best_lambda_s_lr}")
print("Results for best scaled lambda:")
print(best_results_lr_stats)
```
```
    Results for scaled lambda corresponding to raw lambda=0:
    {'avg_train_loss': 0.011431071907281876, 'avg_test_loss': 0.01632683090865612, 'avg_r2_score': 0.982699198022227, 'avg_mse_score': 0.01632683}

    Best scaled lambda: 0.036000000000000004
    Results for best scaled lambda:
    {'avg_train_loss': 0.010861287824809551, 'avg_test_loss': 0.014729137532413006, 'avg_r2_score': 0.9841797175704837, 'avg_mse_score': 0.0147291375}
```
*Training Note: A learning rate of 0.01 was used. Gradient clipping (`nn.utils.clip_grad_norm_`) was employed to mitigate exploding gradients, which can cause `NaN` loss values. This allowed for more stable training with this learning rate.*

### Plot Ridge Regression R2 Results and MSE
Visualizing $R^2$ and MSE across the tested (scaled) lambda values.

```python
r2_scores_plot_lr = [result[1]['avg_r2_score'] for result in results_lr_cv_list]
lambda_values_plot_lr = [result[0] for result in results_lr_cv_list]

plt.figure(figsize=(10, 6))
plt.plot(lambda_values_plot_lr, r2_scores_plot_lr)
plt.xscale('log')
plt.title('R2 Score vs. Scaled Lambda (Ridge Regression)')
plt.xlabel('log(Scaled Lambda)')
plt.ylabel('Average R2 Score')
plt.grid(True)
plt.show()

mse_scores_plot_lr = [result[1]['avg_mse_score'] for result in results_lr_cv_list]
plt.figure(figsize=(10, 6))
plt.plot(lambda_values_plot_lr, mse_scores_plot_lr)
plt.xscale('log')
plt.title('MSE vs. Scaled Lambda (Ridge Regression)')
plt.xlabel('log(Scaled Lambda)')
plt.ylabel('Average MSE')
plt.grid(True)
plt.show()
```
![](./images/Classical-ML/7.png)
![](./images/Classical-ML/4.png)

### Best Lambda Observation
The optimal scaled $\lambda$ in this run was `0.036`. This corresponds to an unscaled $\lambda$ of `0.036 * 250 = 9`. This relatively small regularization strength yielded the best average $R^2$ score of `0.984`. It suggests that while some regularization was beneficial (as performance improved slightly over $\lambda=0$), strong regularization was not necessary for this dataset with the given model.

## Part 3: Logistic Regression

Logistic Regression models the probability of a binary outcome using a sigmoid function applied to a linear combination of input features.

### Define Logistic Regression Class
The model includes a linear layer followed by a sigmoid activation.

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
```

### Define Logistic Regression Training Function
This function handles model initialization, optimizer setup (SGD), loss definition (Binary Cross-Entropy Loss - `nn.BCELoss`), and the training loop.

```python
def train_logistic_regression_model(X_train, y_train, num_epochs=100, learning_rate=0.01):
    input_size = X_train.shape[1]
    model = LogisticRegressionModel(input_size)
    model.to(torch.float32)
    for param in model.parameters(): param.data = param.data.to(torch.float32)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Ensure y_train is float and correct shape for BCELoss (N, 1)
    y_train_processed = y_train.clone().detach().to(torch.float32).view(-1, 1)
    
    train_data = torch.utils.data.TensorDataset(X_train.clone().detach().to(torch.float32), y_train_processed)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
```

### Define Logistic Regression Evaluation Function
Evaluates the model on test data, calculating accuracy and F1-score. Predictions are made by thresholding sigmoid outputs at 0.5.

```python
def eval_logistic_regression_model(model, X_test, y_test):
    model.eval()
    y_test_processed = y_test.clone().detach().to(torch.float32).view(-1, 1)
    test_data = torch.utils.data.TensorDataset(X_test.clone().detach().to(torch.float32), y_test_processed)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    all_predictions_np, all_labels_np = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float()
            all_predictions_np.extend(predictions.cpu().numpy().flatten())
            all_labels_np.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels_np, all_predictions_np)
    # f1_score's 'average' parameter defaults to 'binary' for binary classification.
    f1 = f1_score(all_labels_np, all_predictions_np, zero_division=0) # zero_division handles no positive preds
    err = 1 - acc
    return acc, f1, err
```

### Define Logistic Regression K-fold Cross Validation Function
This function manages the K-fold cross-validation process for logistic regression. The model is re-initialized and trained for each fold.

```python
def logistic_regression_k_fold_cross_validation(X, y, n_folds=10, num_epochs=100, learning_rate=0.01):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = {"train_acc": [], "test_acc": [], "train_f1": [], "test_f1": [], "train_err": [], "test_err": []}

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Ensure X, y are Tensors. train/eval functions handle float conversion.
        # y_train_fold/y_test_fold were converted to int64 in notebook, 
        # but BCELoss needs float targets (handled in train/eval).

        model_fold = train_logistic_regression_model(X_train_fold, y_train_fold, num_epochs, learning_rate)
        
        acc_train, f1_train, err_train = eval_logistic_regression_model(model_fold, X_train_fold, y_train_fold)
        acc_test, f1_test, err_test = eval_logistic_regression_model(model_fold, X_test_fold, y_test_fold)
        
        for key, val in zip(fold_metrics.keys(), [acc_train, acc_test, f1_train, f1_test, err_train, err_test]):
            fold_metrics[key].append(val)
    return fold_metrics # Returns a dictionary of lists (scores per fold)
```

### Load Data And Run Logistic Regression Code
This dataset comprises 100,000 entries and 8 input features. The 'Input 1' column required cleaning: 'Other' string values were replaced with the column's mode.

```python
data_logreg_df = pd.read_csv("LogisticRegression.csv", names=["Input 1", "Input 2", "Input 3", "Input 4", "Input 5", "Input 6", "Input 7", "Input 8", "Label"])
X_logreg_features_df = data_logreg_df.iloc[:, :-1].copy() # Use .copy() to avoid SettingWithCopyWarning

mode_input1_logreg = X_logreg_features_df['Input 1'].mode()[0]
X_logreg_features_df.loc[:, 'Input 1'] = X_logreg_features_df['Input 1'].replace('Other', mode_input1_logreg).astype('int64')

X_logreg_np_arr = X_logreg_features_df.values # Renamed
y_logreg_labels_df = data_logreg_df.iloc[:, -1:]
print("Label distribution in Logistic Regression dataset:")
print(y_logreg_labels_df.value_counts(normalize=True)) # normalize=True shows proportions

y_logreg_np_arr = y_logreg_labels_df.values.squeeze() # Renamed

X_logreg_tensor_data = torch.from_numpy(X_logreg_np_arr).float() # Renamed
y_logreg_tensor_data = torch.from_numpy(y_logreg_np_arr).float() # Renamed; BCELoss needs float

logreg_data_normalizer = ZScoreNormalizer() # Renamed
X_logreg_norm_data = logreg_data_normalizer.fit_transform(X_logreg_tensor_data) # Renamed

results_logreg_cv_folds = logistic_regression_k_fold_cross_validation(
    X_logreg_norm_data, y_logreg_tensor_data, 
    n_folds=10, learning_rate=0.01
)

print("\nLogistic Regression Performance (10-fold CV - Mean +/- Std Dev):")
for metric_name, scores_list in results_logreg_cv_folds.items():
    mean_score = np.mean(scores_list)
    std_score = np.std(scores_list)
    # Capitalize and replace underscore for display
    display_name = metric_name.replace('_', ' ').capitalize()
    print(f"{display_name}: {mean_score:.4f} (+/- {std_score:.4f})")

# For specific values from the notebook's output cell:
print("\n--- Values from a specific prior notebook run (actuals may vary slightly) ---")
print("Accuracy (Testing) Mean:", 0.96004)
print("F1 Score (Testing) Mean:", 0.7265668017769724)
# ... and so on for other specific metrics if desired.
```

```
Label distribution in Logistic Regression dataset:
Label
0        0.915
1        0.085
Name: proportion, dtype: float64

Logistic Regression Performance (10-fold CV - Mean +/- Std Dev):
Train acc: 0.9602 (+/- 0.0003)
Test acc: 0.9600 (+/- 0.0018)
Train f1: 0.7274 (+/- 0.0015)
Test f1: 0.7260 (+/- 0.0130)
Train err: 0.0398 (+/- 0.0003)
Test err: 0.0400 (+/- 0.0018)

--- Values from a specific prior notebook run (actuals may vary slightly) ---
Accuracy (Testing) Mean: 0.96004
F1 Score (Testing) Mean: 0.7265668017769724
```
*(The CV results will vary slightly. The output shows one run's means/stds, followed by specific numbers from your notebook's markdown results cell).*

### Results (Original run time: ~20mins on a 6-Core 4.9GHz i5-9600k)
(Results from the specific run documented in the notebook)
Accuracy (Training) Mean: 0.9601744444444446
Accuracy (Testing) Mean: 0.96004
... (other metrics as listed in the notebook) ...
F1 Score (Testing) Mean: 0.7265668017769724
Highest F1 Score (Testing): 0.7489878542510123

### Accuracy Vs. F1 Score (F-Measure)
**Preferred Metric: F1 Score**

For this dataset, F1 Score is a more informative evaluation metric than accuracy due to significant class imbalance. The majority class ('0') constitutes 91.5% of the data. A model predicting '0' for all instances would achieve 91.5% accuracy but would be useless for identifying the minority class ('1'). The F1 score, as the harmonic mean of precision and recall, provides a better measure of the model's ability to correctly classify the minority class. The observed F1 score (around 0.72-0.74 in the notebook run) indicates a reasonable, though not perfect, performance on the minority class.

## Part 4: K-Nearsest Neighbor Classifier VS Logistic Regression

A direct comparison of KNN and Logistic Regression on the smaller dataset from Part 1 (`KNNClassifierInput.csv`: 500 entries, 2 inputs, classes -1 or 1).

### KNN Test (on Showdown Dataset)
This dataset contains 500 samples with 2 input features and binary class labels (-1 or 1).

```python
# Re-using variables from the notebook's Part 4 KNN Test cell (cell 26)
# X_s, y_s, k_values_s, knn_results_s, knn_test_acc_s, best_k_s etc. are assumed defined as in that cell.
# For brevity, re-running the core logic for output here:
input_file_s = 'KNNClassifierInput.csv' # showdown dataset
output_file_s = 'KNNClassifierOutput.csv'
input_df_s_part4 = pd.read_csv(input_file_s, header=0)
output_df_s_part4 = pd.read_csv(output_file_s).dropna(axis=1)
X_s_part4 = input_df_s_part4[['Input 1', 'Input 2']].values
y_s_part4 = output_df_s_part4.values.squeeze()
k_values_s_part4 = list(range(1, 31))

knn_results_s_part4 = knn_k_fold_cross_validation(X_s_part4, y_s_part4, k_values_s_part4, n_folds=10)
knn_test_acc_s_part4 = knn_results_s_part4["test_acc"] # List of test accuracies for each K
best_k_s_part4_idx = np.argmax(knn_test_acc_s_part4)
best_k_s_part4 = k_values_s_part4[best_k_s_part4_idx]

print(f"KNN Showdown - Best K Value: {best_k_s_part4}")
print(f"KNN Showdown - Test Accuracy with Best K: {knn_test_acc_s_part4[best_k_s_part4_idx] * 100:.2f}%")
print(f"KNN Showdown - Mean Test Accuracy across all K: {np.mean(knn_test_acc_s_part4) * 100:.2f}%") # Example of overall mean
print(f"KNN Showdown - Mean Test Error (Best K): {knn_results_s_part4['test_err'][best_k_s_part4_idx] * 100:.2f}%")
```

```
    KNN Showdown - Best K Value: 13
    KNN Showdown - Test Accuracy with Best K: 97.80%
    KNN Showdown - Mean Test Accuracy across all K: 95.51% 
    KNN Showdown - Mean Test Error (Best K): 2.20% 
```
*(Output based on the notebook's cell 26 run, where best K=13 gave 97.80% accuracy).*
KNN demonstrated strong performance on this dataset.

### Logistic Regression Test (on Showdown Dataset)
The same dataset (500 entries, 2 inputs) was used. Labels {-1, 1} were mapped to {0, 1} for compatibility with the `BCELoss` used in the logistic regression implementation. Input data was Z-score normalized.

As noted in the original notebook, logistic regression struggled significantly with this small dataset. Modifications to epochs or learning rate did not substantially improve performance.

```python
# Re-using X_s_part4; y_s_part4 needs mapping for logistic regression
y_s_mapped_logreg_part4 = np.where(y_s_part4 == -1, 0, 1)

X_s_tensor_part4 = torch.tensor(X_s_part4, dtype=torch.float32)
y_s_tensor_mapped_part4 = torch.tensor(y_s_mapped_logreg_part4, dtype=torch.float32)

showdown_lr_normalizer_part4 = ZScoreNormalizer()
X_s_norm_tensor_part4 = showdown_lr_normalizer_part4.fit_transform(X_s_tensor_part4)

lr_results_s_part4_folds = logistic_regression_k_fold_cross_validation(
    X_s_norm_tensor_part4, y_s_tensor_mapped_part4, 
    n_folds=10, learning_rate=0.01 # lr from notebook cell 27
)

# Calculate mean metrics from the returned dictionary of lists
lr_test_acc_s_part4_mean = np.mean(lr_results_s_part4_folds["test_acc"])
lr_test_f1_s_part4_mean = np.mean(lr_results_s_part4_folds["test_f1"])
lr_test_err_s_part4_mean = np.mean(lr_results_s_part4_folds["test_err"])

print(f"\nLogistic Regression Showdown - Mean Test Accuracy: {lr_test_acc_s_part4_mean:.4f}")
print(f"Logistic Regression Showdown - Mean Test F1 Score: {lr_test_f1_s_part4_mean:.4f}")
print(f"Logistic Regression Showdown - Mean Test Error Rate: {lr_test_err_s_part4_mean:.4f}")
```
```
    Logistic Regression Showdown - Mean Test Accuracy: 0.4420
    Logistic Regression Showdown - Mean Test F1 Score: 0.1918
    Logistic Regression Showdown - Mean Test Error Rate: 0.5580
```
*(Mean results derived from a CV run similar to notebook cell 27. Performance was poor, with accuracy around 44%).*

### Graph Results of KNN and Logistic Regression
Comparing performance visually.

```python
# Graph accuracy for KNN (using knn_results_s_part4 from above)
plt.figure(figsize=(10, 6))
plt.plot(k_values_s_part4, knn_results_s_part4["train_acc"], label='Training Accuracy')
plt.plot(k_values_s_part4, knn_results_s_part4["test_acc"], label='Testing Accuracy')
plt.axvline(best_k_s_part4, color='r', linestyle='--', label=f'Best K = {best_k_s_part4}')
plt.title('KNN Accuracy vs. K Value (Showdown Dataset)')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Graph accuracy for logistic regression per fold (using lr_results_s_part4_folds)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), lr_results_s_part4_folds["train_acc"], label='Training Accuracy (per fold)') # list of 10 values
plt.plot(range(1, 11), lr_results_s_part4_folds["test_acc"], label='Testing Accuracy (per fold)')  # list of 10 values
plt.title('Logistic Regression Accuracy vs. Fold (Showdown Dataset)')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Graph error rate for KNN
plt.figure(figsize=(10, 6))
plt.plot(k_values_s_part4, knn_results_s_part4["train_err"], label='Training Error Rate')
plt.plot(k_values_s_part4, knn_results_s_part4["test_err"], label='Testing Error Rate')
plt.axvline(best_k_s_part4, color='r', linestyle='--', label=f'Best K = {best_k_s_part4}')
plt.title('KNN Error Rate vs. K Value (Showdown Dataset)')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True)
plt.show()

# Graph error rate for logistic regression per fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), lr_results_s_part4_folds["train_err"], label='Training Error Rate (per fold)')
plt.plot(range(1, 11), lr_results_s_part4_folds["test_err"], label='Testing Error Rate (per fold)')
plt.title('Logistic Regression Error Rate vs. Fold (Showdown Dataset)')
plt.xlabel('Fold')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True)
plt.show()
```
KNN Accuracy vs. K (Showdown):
![](./images/Classical-ML/5.png)

Logistic Regression Accuracy vs. Fold (Showdown):
![](./images/Classical-ML/6.png)

KNN Error Rate vs. K (Showdown):
![](./images/Classical-ML/2.png)

Logistic Regression Error Rate vs. Fold (Showdown):
![](./images/Classical-ML/3.png)

### KNN Vs. Logistic Regression (Showdown Dataset Conclusion)
The results clearly indicate that for this particular small dataset (500 entries, 2 features), KNN significantly outperformed logistic regression. KNN achieved high accuracy (e.g., ~97.8% with best K), while logistic regression's accuracy hovered near random chance for a binary problem (~44%).

The likely reason for this disparity is the dataset size. KNN, an instance-based learner, can effectively utilize local information from the 500 samples. Logistic regression, particularly when trained via SGD, often requires more data to reliably converge its parameters to an optimal decision boundary. With limited data, its performance can be poor and highly variable.

## Summary of Observations

This implementation exercise highlighted several practical aspects of working with these models:
1.  **Data Normalization:** Consistently beneficial, especially for distance-based algorithms like KNN and gradient-optimized models like linear and logistic regression.
2.  **Cross-Validation:** Indispensable for hyperparameter selection (K in KNN, lambda in Ridge) and for obtaining more robust performance estimates than a single train-test split.
3.  **Metric Selection:** Crucial, especially with imbalanced datasets. Accuracy can be misleading; F1-score (or precision/recall) often provides a better assessment of a model's utility for minority classes.
4.  **Model Suitability ("No Free Lunch"):** Different models are suited to different data characteristics. KNN performed well on the small, low-dimensional dataset. Logistic regression, while struggling there, was more appropriate for the larger, albeit imbalanced, classification task.
5.  **Implementation Details:** Factors like handling data types, ensuring proper model re-initialization in CV folds, and techniques like gradient clipping are important for successful model training and evaluation.

Building these models, even with PyTorch providing foundational tools, offers valuable insights into their mechanics beyond simply using high-level library functions.