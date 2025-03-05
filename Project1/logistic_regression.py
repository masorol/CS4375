import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegressionL2:
    def __init__(self, learning_rate=0.1, max_iter=100, l2=0.1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        X = np.hstack((np.ones((n_samples, 1)), X))

        for _ in range(self.max_iter):
            z = X @ self.weights
            h = self.sigmoid(z)
            gradient = X.T @ (h - y) / n_samples 
            
            gradient[1:] += self.l2 * self.weights[1:]
            
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return (self.sigmoid(X @ self.weights) >= 0.5).astype(int)

def tune_hyperparameter(X, y, l2_values, learning_rate=0.1, max_iter=100):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    best_l2 = None
    best_f1 = -1

    for l2 in l2_values:
        model = LogisticRegressionL2(learning_rate=learning_rate, max_iter=max_iter, l2=l2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, pos_label=1)

        if f1 > best_f1:
            best_f1 = f1
            best_l2 = l2

    return best_l2


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return accuracy, precision, recall, f1
