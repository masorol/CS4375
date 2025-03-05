import sys
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from multinomial_naive_bayes import MultinomialNaiveBayes, evaluate_model as mnb_evaluate
from bernoulli_naive_bayes import BernoulliNaiveBayes, evaluate_model as bnb_evaluate
from logistic_regression import LogisticRegressionL2, tune_hyperparameter, evaluate_model as lr_evaluate

nltk.download('punkt')
nltk.download('stopwords')

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([token for token in tokens if token not in stop_words])

def read_files(directory):
    texts = []
    labels = []
    for label in ['spam', 'ham']:
        path = os.path.join(directory, label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
                processed_text = preprocess_text(raw_text)
                texts.append(processed_text)
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
    full_path = os.path.join(output_dir, filename)
    df.to_csv(full_path, index=False)


def main(model_choice):
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

        vectorizer = CountVectorizer()
        vectorizer.fit(train_texts)
        X_test = vectorizer.transform(test_texts)

        if model_choice == 'mnb':
            X_train, y_train = create_feature_matrix(train_texts, train_labels, vectorizer)
            X_test, y_test = create_feature_matrix(test_texts, test_labels, vectorizer)
            
            save_to_csv(X_train, y_train, f'enron{i}_bow_train.csv')
            save_to_csv(X_test, y_test, f'enron{i}_bow_test.csv')
            
            model = MultinomialNaiveBayes()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy, precision, recall, f1 = mnb_evaluate(y_test, predictions)
            print(f"Enron{i} Multinomial Naive Bayes Results:")
        
        elif model_choice == 'bnb':
            X_train, y_train = create_feature_matrix(train_texts, train_labels, vectorizer, is_bernoulli=True)
            X_test, y_test = create_feature_matrix(test_texts, test_labels, vectorizer, is_bernoulli=True)
            
            save_to_csv(X_train, y_train, f'enron{i}_bernoulli_train.csv')
            save_to_csv(X_test, y_test, f'enron{i}_bernoulli_test.csv')
            
            model = BernoulliNaiveBayes()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy, precision, recall, f1 = bnb_evaluate(y_test, predictions)
            print(f"Enron{i} Bernoulli Naive Bayes Results:")
        
        elif model_choice == 'lr':
            l2_values = [0.01, 0.1, 1]
            
            for rep_name, is_bernoulli in [('BoW', False), ('Bernoulli', True)]:
                X_train, y_train = create_feature_matrix(train_texts, train_labels, vectorizer, is_bernoulli)
                X_test, y_test = create_feature_matrix(test_texts, test_labels, vectorizer, is_bernoulli)
                
                X_train_dense = X_train.toarray()
                X_test_dense = X_test.toarray()
                
                best_l2 = tune_hyperparameter(X_train_dense, y_train, l2_values)
                
                model = LogisticRegressionL2(l2=best_l2, max_iter=100, learning_rate=0.1)
                model.fit(X_train_dense, y_train)
                
                predictions = model.predict(X_test_dense)
                accuracy, precision, recall, f1 = lr_evaluate(y_test, predictions)
                
                print(f"\nEnron{i} Logistic Regression ({rep_name}) Results:")
                print(f"Best L2: {best_l2}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-score: {f1:.4f}")
        
        
        if model_choice in ['mnb', 'bnb']:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"Datasets for enron{i} have been saved.")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['mnb', 'bnb', 'lr']:
        print("Invalid model choice. Please call: python main.py [mnb|bnb|lr]")
        sys.exit(1)
    main(sys.argv[1])

