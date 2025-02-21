import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

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

print("Processing completed.")
