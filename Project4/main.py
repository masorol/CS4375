import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=10000, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'linear': {
        'kernel': 'linear',
        'C_values': [0.1, 1, 10],
    },
    'poly': {
        'kernel': 'poly',
        'degree_values': [2, 3, 4],
        'C_values': [0.1, 1, 10],
    },
    'rbf': {
        'kernel': 'rbf',
        'gamma_values': ['scale', 0.001, 0.01, 0.1],
        'C_values': [0.1, 1, 10],
    }
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = []

for model_name, params in models.items():
    print(f"\nEvaluating {model_name} kernel")
    
    if model_name == 'linear':
        for C in params['C_values']:
            svm = SVC(kernel=params['kernel'], C=C, random_state=42)
            scores = cross_val_score(svm, X_train, y_train, cv=cv, n_jobs=-1)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results.append({
                'kernel': params['kernel'],
                'C': C,
                'degree': '-',
                'gamma': '-',
                'mean_accuracy': mean_score,
                'std_accuracy': std_score
            })
    elif model_name == 'poly':
        for degree in params['degree_values']:
            for C in params['C_values']:
                svm = SVC(kernel=params['kernel'], degree=degree, C=C, random_state=42)
                scores = cross_val_score(svm, X_train, y_train, cv=cv, n_jobs=-1)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                results.append({
                    'kernel': params['kernel'],
                    'C': C,
                    'degree': degree,
                    'gamma': '-',
                    'mean_accuracy': mean_score,
                    'std_accuracy': std_score
                })
    elif model_name == 'rbf':
        for gamma in params['gamma_values']:
            for C in params['C_values']:
                svm = SVC(kernel=params['kernel'], gamma=gamma, C=C, random_state=42)
                scores = cross_val_score(svm, X_train, y_train, cv=cv, n_jobs=-1)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                results.append({
                    'kernel': params['kernel'],
                    'C': C,
                    'degree': '-',
                    'gamma': gamma,
                    'mean_accuracy': mean_score,
                    'std_accuracy': std_score
                })

print("\nCross-validation results:")
print("{:<10} {:<5} {:<6} {:<10} {:<15} {:<15}".format(
    'Kernel', 'C', 'Degree', 'Gamma', 'Mean Accuracy', 'Std Accuracy'))
print("-" * 70)

printed_configs = set()

for res in sorted(results, key=lambda x: -x['mean_accuracy']):
    config_key = (res['kernel'], res['C'], res['degree'], str(res['gamma']))
    if config_key not in printed_configs:
        printed_configs.add(config_key)
        print("{:<10} {:<5} {:<6} {:<10} {:<15.4f} {:<15.4f}".format(
            res['kernel'],
            res['C'],
            res['degree'],
            str(res['gamma']),
            res['mean_accuracy'],
            res['std_accuracy']
        ))

best_model = max(results, key=lambda x: x['mean_accuracy'])
print(f"\nBest model configuration:")
print(f"Kernel: {best_model['kernel']}")
print(f"C: {best_model['C']}")
print(f"Degree: {best_model['degree']}")
print(f"Gamma: {best_model['gamma']}")
print(f"Validation Accuracy: {best_model['mean_accuracy']:.4f} ± {best_model['std_accuracy']:.4f}")

if best_model['kernel'] == 'linear':
    final_model = SVC(kernel=best_model['kernel'], C=best_model['C'], random_state=42)
elif best_model['kernel'] == 'poly':
    final_model = SVC(kernel=best_model['kernel'], degree=best_model['degree'], 
                      C=best_model['C'], random_state=42)
else:
    final_model = SVC(kernel=best_model['kernel'], gamma=best_model['gamma'], 
                      C=best_model['C'], random_state=42)

final_model.fit(X_train, y_train)
test_accuracy = final_model.score(X_test, y_test)
print(f"\nTest accuracy of best model: {test_accuracy:.4f}")