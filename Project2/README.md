# Project 2: Evaluation of Tree-Based Classifiers and Their Ensembles

This project evaluates various tree-based classifiers and their ensemble methods on both custom CSV datasets and the MNIST dataset.

## Requirements

- Python 3.9+
- Required packages:
  ```bash
  pip install numpy pandas scikit-learn
  ```

## File Structure

Before running, please organize your files like so:

```
project/
├── datasets/
│ ├── test_c300_d100
│ ├── test_c300_d1000
│ ├── test_c300_d5000
│ ├── test_c500_d100
│ ├── test_c500_d1000
│ ├── test_c500_d5000
│ ├── test_c1000_d100
│ ├── test_c1000_d1000
│ ├── test_c1000_d5000
│ ├── test_c1500_d100
│ ├── test_c1500_d1000
│ ├── test_c1500_d5000
│ ├── test_c1800_d100
│ ├── test_c1800_d1000
│ ├── test_c1800_d5000
│ ├── train_... (remaining training files)
│ └── valid_...  (remaining validation files)
└── main.py
```

## Usage

Run the script using:

```
python main.py
```

### Dataset Selection:

- Choose between 'csv' for custom datasets or 'mnist' for the MNIST dataset.

### Model Selection (CSV only):

- DT: Decision Trees
- Bagging: Bagging Classifier
- RF: Random Forest
- GB: Gradient Boosting

### MNIST Evaluation:

- Automatically evaluates all four classifiers on the MNIST dataset.

## Results

The console output will display:

- Best hyperparameter settings found via tuning
- Classification accuracy
- F1 scores (for CSV datasets only)
