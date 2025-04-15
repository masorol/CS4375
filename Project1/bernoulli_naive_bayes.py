import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_neg_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(class_counts) - np.log(len(y))

        n_features = X.shape[1]
        feature_counts = np.zeros((len(self.classes_), n_features))
        for idx, c in enumerate(self.classes_):
            if issparse(X):
                feature_counts[idx, :] = X[y == c].sum(axis=0).A1
            else:
                feature_counts[idx, :] = X[y == c].sum(axis=0)
        
        smoothed_fc = feature_counts + 1
        smoothed_totals = class_counts.reshape(-1, 1) + 2
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_totals)
        self.feature_log_neg_prob_ = np.log(smoothed_totals - smoothed_fc) - np.log(smoothed_totals)

    def predict(self, X):
        if issparse(X):
            X = X.toarray()
        log_probs = (X @ self.feature_log_prob_.T + 
                     (1 - X) @ self.feature_log_neg_prob_.T + 
                     self.class_log_prior_)
        return self.classes_[np.argmax(log_probs, axis=1)]

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return accuracy, precision, recall, f1
