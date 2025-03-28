import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
from itertools import product

def read_dataset(dataset_name, path):
    return pd.read_csv(os.path.join(path, dataset_name), header=None)

def split_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def tune_and_evaluate(train_data, val_data, test_data, compute_f1=True):
    X_train, y_train = split_data(train_data)
    X_val, y_val = split_data(val_data)
    X_test, y_test = split_data(test_data)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15]
    }

    best_score = -1
    best_params = None
    
    keys, values = zip(*param_grid.items())
    for params in product(*values):
        current_params = dict(zip(keys, params))
        dt = DecisionTreeClassifier(**current_params, random_state=42)
        dt.fit(X_train, y_train)
        val_score = dt.score(X_val, y_val)
        
        if val_score > best_score:
            best_score = val_score
            best_params = current_params

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    final_model = DecisionTreeClassifier(**best_params, random_state=42)
    final_model.fit(X_combined, y_combined)
    
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if compute_f1:
        f1 = f1_score(y_test, y_pred, average='macro')
    else:
        f1 = None
    
    return best_params, accuracy, f1

def tune_and_evaluate_bagging(train_data, val_data, test_data, compute_f1=True):
    X_train, y_train = split_data(train_data)
    X_val, y_val = split_data(val_data)
    X_test, y_test = split_data(test_data)

    param_grid = {
        'n_estimators': [10, 50],
        'max_samples': [0.5, 0.8, 1.0],
        'base_criterion': ['gini', 'entropy'],
        'base_max_depth': [None, 5, 10, 15],
        'base_splitter': ['best', 'random']
    }

    best_score = -1
    best_params = None

    keys, values = zip(*param_grid.items())
    for params in product(*values):
        current_params = dict(zip(keys, params))
        
        base_dt = DecisionTreeClassifier(
            criterion=current_params.pop('base_criterion'),
            max_depth=current_params.pop('base_max_depth'),
            splitter=current_params.pop('base_splitter'),
            random_state=42
        )
        
        bagging = BaggingClassifier(
            estimator=base_dt,
            **current_params,
            random_state=42
        )
        
        bagging.fit(X_train, y_train)
        val_score = bagging.score(X_val, y_val)
        
        if val_score > best_score:
            best_score = val_score
            best_params = {
                'n_estimators': current_params['n_estimators'],
                'max_samples': current_params['max_samples'],
                'criterion': base_dt.criterion,
                'max_depth': base_dt.max_depth,
                'splitter': base_dt.splitter
            }

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    final_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            criterion=best_params['criterion'],
            max_depth=best_params['max_depth'],
            splitter=best_params['splitter'],
            random_state=42
        ),
        n_estimators=best_params['n_estimators'],
        max_samples=best_params['max_samples'],
        random_state=42
    )
    final_model.fit(X_combined, y_combined)
    
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if compute_f1:
        f1 = f1_score(y_test, y_pred, average='macro')
    else:
        f1 = None

    return best_params, accuracy, f1

def tune_and_evaluate_rf(train_data, val_data, test_data, compute_f1=True):
    X_train, y_train = split_data(train_data)
    X_val, y_val = split_data(val_data)
    X_test, y_test = split_data(test_data)

    param_grid = {
        'n_estimators': [50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'max_features': ['sqrt', 0.5],
        'min_samples_split': [2, 5]
    }

    best_score = -1
    best_params = None

    keys, values = zip(*param_grid.items())
    for params in product(*values):
        current_params = dict(zip(keys, params))
        
        rf = RandomForestClassifier(
            **current_params,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        val_score = rf.score(X_val, y_val)
        
        if val_score > best_score:
            best_score = val_score
            best_params = current_params

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_combined, y_combined)
    
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if compute_f1:
        f1 = f1_score(y_test, y_pred, average='macro')
    else:
        f1 = None

    return best_params, accuracy, f1

def tune_and_evaluate_gb(train_data, val_data, test_data, compute_f1=True):
    X_train, y_train = split_data(train_data)
    X_val, y_val = split_data(val_data)
    X_test, y_test = split_data(test_data)

    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }

    best_score = -1
    best_params = None

    keys, values = zip(*param_grid.items())
    for params in product(*values):
        current_params = dict(zip(keys, params))
        
        gb = GradientBoostingClassifier(
            **current_params,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        val_score = gb.score(X_val, y_val)
        
        if val_score > best_score:
            best_score = val_score
            best_params = current_params

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    final_model = GradientBoostingClassifier(**best_params, random_state=42)
    final_model.fit(X_combined, y_combined)
    
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if compute_f1:
        f1 = f1_score(y_test, y_pred, average='macro')
    else:
        f1 = None
        
    return best_params, accuracy, f1

def main():
    script_path = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(script_path, 'datasets')

    dataset_choice = input("Choose dataset type (csv/mnist): ").strip().lower()
    while dataset_choice not in ['csv', 'mnist']:
        print("Invalid choice. Please enter 'csv' or 'mnist'")
        dataset_choice = input("Choose dataset type (csv/mnist): ").strip().lower()

    if dataset_choice == 'csv':
        model_type = input("Choose model type (dt/bagging/rf/gb): ").strip().lower()
        while model_type not in ['dt', 'bagging', 'rf', 'gb']:
            print("Invalid choice. Please enter 'dt', 'bagging', 'rf' or 'gb'")
            model_type = input("Choose model type (dt/bagging/rf/gb): ").strip().lower()

        for filename in os.listdir(datasets_path):
            if filename.startswith('train'):
                parts = filename.split('_')
                clauses = parts[1][1:]
                examples = parts[2][1:-4]

                train_data = read_dataset(filename, datasets_path)
                val_filename = f'valid_c{clauses}_d{examples}.csv'
                test_filename = f'test_c{clauses}_d{examples}.csv'
                
                val_data = read_dataset(val_filename, datasets_path)
                test_data = read_dataset(test_filename, datasets_path)

                if model_type == 'dt':
                    best_params, accuracy, f1 = tune_and_evaluate(train_data, val_data, test_data)
                elif model_type == 'bagging':
                    best_params, accuracy, f1 = tune_and_evaluate_bagging(train_data, val_data, test_data)
                elif model_type == 'rf':
                    best_params, accuracy, f1 = tune_and_evaluate_rf(train_data, val_data, test_data)
                else:
                    best_params, accuracy, f1 = tune_and_evaluate_gb(train_data, val_data, test_data)

                print(f"\nDataset: {filename}")
                print(f"Best Hyperparameters: {best_params}")
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Test F1 Score: {f1:.4f}")
    else:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split

        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X / 255.0
        y = y.astype(int)

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=10000, random_state=42
        )

        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train.values
        val_data = pd.DataFrame(X_val)
        val_data['target'] = y_val.values
        test_data = pd.DataFrame(X_test)
        test_data['target'] = y_test.values

        classifiers = [
            ('Decision Tree', tune_and_evaluate),
            ('Bagging', tune_and_evaluate_bagging),
            ('Random Forest', tune_and_evaluate_rf),
            ('Gradient Boosting', tune_and_evaluate_gb)
        ]

        for name, classifier_func in classifiers:
            print(f"\nEvaluating {name} on MNIST")
            best_params, accuracy, _ = classifier_func(
                train_data, val_data, test_data, compute_f1=False
            )
            print(f"Best Parameters: {best_params}")
            print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()