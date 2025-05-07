# Project 4: Support Vector Machines (SVMs)

This project evaluates different SVM kernels (linear, polynomial, and RBF) with various hyperparameter configurations on the MNIST dataset. It performs systematic hyperparameter evaluation using 3-fold cross-validation and reports the best performing model configuration.

## Requirements
- Python 3.6+
- Required packages:
  ```bash
  pip install numpy scikit-learn matplotlib
  ```

## Usage
Run the script using:
```
python main.py
```

### Evaluation Process:
- Tests three SVM kernels
- Uses stratified 3-fold cross-validation
- Reports mean and standard deviation of accuracy for each configuration

## Results
The console output will display:
- All evaluated configurations sorted by validation accuracy
- Best model configuration with validation metrics
- Final test accuracy of the best model
