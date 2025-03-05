# Project 1: Email Classification

## Requirements

- Python 3.9+
- Required packages:
  ```bash
  pip install numpy pandas scikit-learn nltk
  ```

## File Structure

Before running, please organize your files like so:

```
project/
├── datasets/
│ ├── enron1/
│ │ ├── train/
│ │ │ ├── spam/
│ │ │ └── ham/
│ │ ├── test/
│ │ │ ├── spam/
│ │ │ └── ham/
│ ├── enron2/...
│ └── enron4/...
├── output/
├── main.py
├── bernoulli_naive_bayes.py
├── logistic_regression.py
└── multinomial_naive_bayes.py
```

## Usage

### Run models (one at a time):

#### Multinomial Naive Bayes (Bag-of-Words):
python main.py mnb
#### Bernoulli Naive Bayes:
python main.py bnb
#### Logistic Regression:
python main.py lr

### Find results:

- Generated CSV files will appear in /output
- Console output will show accuracy, precision, recall, and F1-scores
