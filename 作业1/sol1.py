import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the training and test datasets
train = pd.read_csv('data/simreg1_train.csv', sep=';')
test = pd.read_csv('data/simreg1_test.csv', sep=';')

# linear regression
X = np.array([train.iloc[:, 0]]).T
y_train = train.iloc[:, 1]
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y_train
print("beta hat: ", beta_hat)

# predict on X and plot the result
y_hat = X @ beta_hat
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(X, y_train, 'o', label='Actual')
plt.plot(X, y_hat, label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')

# Calculate training error
error = np.mean((y_hat - y_train) ** 2)
print("Training error: ", error)

# Calculate training error for test dataset
X_test = np.array([test.iloc[:, 0]]).T
y_test = test.iloc[:, 1]
y_hat_test = X_test @ beta_hat
error_test = np.mean((y_hat_test - y_test) ** 2)
print("Test error: ", error_test)

def calculate_errors(train, test, n):
    """
    Calculate the training and test errors for different polynomial degrees from 1 to n.

    Parameters:
    train (pd.DataFrame): Training dataset.
    test (pd.DataFrame): Test dataset.
    n (int): Maximum polynomial degree.

    Returns:
    pd.DataFrame: DataFrame containing the training and test errors for each polynomial degree.
    """
    errors = []
    for k0 in range(1, n+1):
        X = np.ones((len(train), 1))
        for k in range(1, k0 + 1):
            X = np.column_stack((X, train.iloc[:, 0] ** k))
        y = train.iloc[:, 1]
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        y_hat = X @ beta_hat

        new_error = {'k': k0, 'train_error' : np.mean((y_hat - y) ** 2)}

        X_test = np.ones((len(test), 1))
        for k in range(1, k0 + 1):
            X_test = np.column_stack((X_test, test.iloc[:, 0] ** k))
        y_test = test.iloc[:, 1]
        y_hat_test = X_test @ beta_hat
        new_error['test_error'] = np.mean((y_hat_test - y_test) ** 2)
        errors.append(new_error)
        
    return pd.DataFrame(errors)

errors = calculate_errors(train, test, 8)
plt.subplot(1,2,2)
plt.title('Polynomial Regression Error vs Degree')
plt.xlabel('k')
plt.ylabel('Error')
plt.plot(errors['k'], errors['test_error'], 'o-', label='test_error')
plt.plot(errors['k'], errors['train_error'], 'o-', label='train_error')
plt.legend()
plt.show()