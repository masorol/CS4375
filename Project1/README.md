# Email Classification Project

## Requirements

- Python 3.9+
- Required packages:
  ```bash
  pip install numpy pandas scikit-learn nltk
  ```

## File Structure

Before running, please organize your files like so:

project/
├── datasets/
│ ├── enron1/
│ │ ├── train/
│ │ │ ├── spam/
│ │ │ └── ham/
│ │ └── test/
│ │ ├── spam/
│ │ └── ham/
│ ├── enron2/...
│ └── enron4/...
├── output/
├── main.py
├── bernoulli_naive_bayes.py
├── logistic_regression.py
└── multinomial_naive_bayes.py

## Usage

### Run models (one at a time):

python main.py mnb # Multinomial Naive Bayes (BoW)
python main.py bnb # Bernoulli Naive Bayes
python main.py lr # Logistic Regression

### Find results:

- Generated CSV files will appear in /output
- Console output will show accuracy, precision, recall, and F1-scores
