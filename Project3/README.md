# Project 3: Neural Network Architecture Evaluation on MNIST and CIFAR-10

This project implements and compares various neural network architectures (MLPs and CNNs) on MNIST and CIFAR-10 datasets. It performs hyperparameter tuning via random search, evaluates models with 3-fold cross-validation, and reports final test accuracy.

## Requirements
- Python 3.8+
- Required packages:
  ```bash
  pip install torch torchvision numpy matplotlib scikit-learn
  ```

## Usage
Run the script using:
```
python main.py
```

### Dataset Selection:
- When prompted, choose 'MNIST' or 'CIFAR10'.

### Model Selection:
- When prompted, choose 'MLP' or 'CNN'.

### MNIST Evaluation:
- Loads the dataset with appropriate normalization
- Performs hyperparameter search (learning rate, batch size, optimizer, dropout) via random sampling
- Runs 3-fold cross-validation (5 epochs per fold during tuning)
- Selects the best hyperparameters based on validation accuracy
- Trains the final model on the entire training set for 15 epochs with the best hyperparameters
- Evaluates final accuracy on the held-out test set

## Results
The console output will display:
- Best hyperparameters found
- Cross-validation mean ± std accuracy
- Final test accuracy
