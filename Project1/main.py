import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_files(directory):
    texts = []
    labels = []
    for label in ['spam', 'ham']:
        path = os.path.join(directory, label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read().lower())
                labels.append(label)
    return texts, labels

def create_feature_matrix(texts, labels, vectorizer, is_bernoulli=False):
    X = vectorizer.transform(texts)
    if is_bernoulli:
        X = (X > 0).astype(int)
    y = LabelEncoder().fit_transform(labels)
    return X, y

def save_to_csv(X, y, filename):
    df = pd.DataFrame(X.toarray())
    df['label'] = y
    df.to_csv(filename, index=False)

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(class_counts) - np.log(len(y))

        feature_counts = np.zeros((len(self.classes_), X.shape[1]))
        for idx, c in enumerate(self.classes_):
            feature_counts[idx, :] = X[y == c].sum(axis=0)
        
        smoothed_fc = feature_counts + 1  # Add-one Laplace smoothing
        smoothed_totals = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_totals)

    def predict(self, X):
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(log_probs, axis=1)]

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(class_counts) - np.log(len(y))

        feature_counts = np.zeros((len(self.classes_), X.shape[1]))
        for idx, c in enumerate(self.classes_):
            feature_counts[idx, :] = X[y == c].sum(axis=0)
        
        smoothed_fc = feature_counts + 1  # Add-one Laplace smoothing
        smoothed_totals = class_counts.reshape(-1, 1) + 2  # Total count for each class + 2
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_totals)
        self.feature_log_neg_prob_ = np.log(smoothed_totals - smoothed_fc) - np.log(smoothed_totals)

    def predict(self, X):
        log_probs = (X @ self.feature_log_prob_.T + 
                     (1 - X) @ self.feature_log_neg_prob_.T + 
                     self.class_log_prior_)
        return self.classes_[np.argmax(log_probs, axis=1)]

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_dir, 'datasets')

for i in [1, 2, 4]:
    train_dir = os.path.join(base_path, f'enron{i}', 'train')
    test_dir = os.path.join(base_path, f'enron{i}', 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"enron{i} not found.")
        continue

    print(f"\nProcessing enron{i}")

    train_texts, train_labels = read_files(train_dir)
    test_texts, test_labels = read_files(test_dir)

    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(train_texts)

    X_train_bow, y_train = create_feature_matrix(train_texts, train_labels, vectorizer)
    X_test_bow, y_test = create_feature_matrix(test_texts, test_labels, vectorizer)

    save_to_csv(X_train_bow, y_train, f'enron{i}_bow_train.csv')
    save_to_csv(X_test_bow, y_test, f'enron{i}_bow_test.csv')

    X_train_bernoulli, _ = create_feature_matrix(train_texts, train_labels, vectorizer, is_bernoulli=True)
    X_test_bernoulli, _ = create_feature_matrix(test_texts, test_labels, vectorizer, is_bernoulli=True)

    save_to_csv(X_train_bernoulli, y_train, f'enron{i}_bernoulli_train.csv')
    save_to_csv(X_test_bernoulli, y_test, f'enron{i}_bernoulli_test.csv')

    print(f"enron{i} processed and saved.")

    # Train and evaluate Multinomial Naive Bayes model
    mnb_model = MultinomialNaiveBayes()
    mnb_model.fit(X_train_bow, y_train)
    mnb_predictions = mnb_model.predict(X_test_bow)

    mnb_accuracy, mnb_precision, mnb_recall, mnb_f1 = evaluate_model(y_test, mnb_predictions)

    print(f"Enron{i} Multinomial Naive Bayes Results:")
    print(f"Accuracy: {mnb_accuracy:.4f}")
    print(f"Precision: {mnb_precision:.4f}")
    print(f"Recall: {mnb_recall:.4f}")
    print(f"F1-score: {mnb_f1:.4f}")

    # Train and evaluate Bernoulli Naive Bayes model
    bnb_model = BernoulliNaiveBayes()
    bnb_model.fit(X_train_bernoulli, y_train)
    bnb_predictions = bnb_model.predict(X_test_bernoulli)

    bnb_accuracy, bnb_precision, bnb_recall, bnb_f1 = evaluate_model(y_test, bnb_predictions)

    print(f"Enron{i} Bernoulli Naive Bayes Results:")
    print(f"Accuracy: {bnb_accuracy:.4f}")
    print(f"Precision: {bnb_precision:.4f}")
    print(f"Recall: {bnb_recall:.4f}")
    print(f"F1-score: {bnb_f1:.4f}")

print("Processing completed.")
