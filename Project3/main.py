import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
import random
import matplotlib.pyplot as plt


def load_mnist(normalization_type='standard'):
    if normalization_type == 'minmax':
        transform = transforms.Compose([transforms.ToTensor()])
    elif normalization_type == 'standard':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_set, test_set

def load_cifar10(normalization_type='standard'):
    if normalization_type == 'minmax':
        transform = transforms.Compose([transforms.ToTensor()])
    elif normalization_type == 'standard':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_set, test_set

def random_search_hyperparameters():
    return {
        'batch_size': random.choice([64, 128]),
        'lr': random.choice([1e-2, 1e-3]),
        'optimizer': random.choice(['adam', 'sgd']),
        'dropout': random.choice([0.3, 0.5])
    }


def train_with_hyperparams(train_set, model_class, input_params, hyperparams, is_cnn=False, num_epochs=15):
    train_loader = DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True)
    
    if is_cnn:
        model = model_class(input_params['in_channels'], 10, hyperparams.get('dropout', 0))
    else:
        model = model_class(input_params['input_size'], 10)
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr']) if hyperparams['optimizer'] == 'adam' \
        else optim.SGD(model.parameters(), lr=hyperparams['lr'], momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
    
    return model


def train_and_evaluate(dataset, model_class, hyperparams, is_cnn=False, num_epochs=5):
    kfold = KFold(n_splits=3, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {
        'final_acc': [],
        'train_losses': [],
        'val_accuracies': []
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n=== Fold {fold+1}/{3} ===")
        
        train_loader = DataLoader(dataset, batch_size=hyperparams['batch_size'],
                                sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(dataset, batch_size=hyperparams['batch_size'],
                              sampler=SubsetRandomSampler(val_idx))

        model = model_class().to(device)
        
        if hyperparams['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=hyperparams['lr'], momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        fold_train_loss = []
        fold_val_acc = []
        
        best_val = 0
        patience = 2
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            fold_train_loss.append(avg_loss)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            fold_val_acc.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            if val_acc > best_val:
                best_val = val_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"No improvement for {patience} epochs, stopping early.")
                    break

        results['final_acc'].append(fold_val_acc[-1])
        results['train_losses'].append(fold_train_loss)
        results['val_accuracies'].append(fold_val_acc)

    cv_mean = np.mean(results['final_acc'])
    cv_std = np.std(results['final_acc'])
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_loss_history': results['train_losses'],
        'val_acc_history': results['val_accuracies']
    }

def evaluate(model, dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


class FastMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class MediumMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class DeepMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

class FastCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class EnhancedCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class DeepCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    

def hyperparameter_tuning(dataset, model_class, input_params, is_cnn=False):
    best_acc = 0
    best_hyperparams = {}
    best_histories = None
    all_cv_scores = []
    
    for _ in range(5):
        hyperparams = random_search_hyperparameters()
        print(f"\nTesting {hyperparams}")
        
        if is_cnn:
            model_wrapper = lambda: model_class(
                input_params['in_channels'], 
                10, 
                hyperparams['dropout']
            )
        else:
            model_wrapper = lambda: model_class(
                input_params['input_size'], 
                10, 
                hyperparams['dropout']
            )
        
        result = train_and_evaluate(
            dataset=dataset,
            model_class=model_wrapper,
            hyperparams=hyperparams,
            is_cnn=is_cnn
        )
        
        all_cv_scores.append(result['cv_mean'])
        
        if result['cv_mean'] > best_acc:
            best_acc = result['cv_mean']
            best_hyperparams = hyperparams
            best_histories = {
                'train_loss': result['train_loss_history'],
                'val_acc': result['val_acc_history']
            }
    
    final_std = np.std(all_cv_scores)
    
    return best_hyperparams, {
        'cv_mean': best_acc,
        'cv_std': final_std,
        'train_loss': best_histories['train_loss'],
        'val_acc': best_histories['val_acc']
    }

def plot_training_curves(train_losses, val_accuracies, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title(f"{model_name} - Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f"{model_name} - Validation Accuracy")
    plt.show()


def main():
    dataset_choice = input("Choose dataset (MNIST/CIFAR10): ").lower()
    model_type = input("Choose model type (MLP/CNN): ").lower()
    
    if dataset_choice == "mnist":
        train_set, test_set = load_mnist(normalization_type='standard')
        input_params = {'input_size': 28*28, 'in_channels': 1}
    elif dataset_choice == "cifar10":
        train_set, test_set = load_cifar10(normalization_type='standard')
        input_params = {'input_size': 3*32*32, 'in_channels': 3}

    if model_type == "mlp":
        models_dict = {
            "Shallow MLP": FastMLP,
            "Medium MLP": MediumMLP,
            "Deep MLP": DeepMLP
        }
    elif model_type == "cnn":
        models_dict = {
            "Base CNN": FastCNN,
            "Enhanced CNN": EnhancedCNN,
            "Deep CNN": DeepCNN
        }

    for model_name, model_class in models_dict.items():
            print(f"\n=== Evaluating {model_name} ===")
            
            best_hyperparams, best_histories = hyperparameter_tuning(
                dataset=train_set,
                model_class=model_class,
                input_params=input_params,
                is_cnn=(model_type == "cnn")
            )
            
            avg_train_loss = np.mean(best_histories['train_loss'], axis=0)
            avg_val_acc = np.mean(best_histories['val_acc'], axis=0)
            plot_training_curves(avg_train_loss, avg_val_acc, model_name)
            
            final_model = train_with_hyperparams(
                train_set=train_set,
                model_class=model_class,
                input_params=input_params,
                hyperparams=best_hyperparams,
                is_cnn=(model_type == "cnn"),
                num_epochs=15
            )
            
            test_acc = evaluate(final_model, test_set)
            
            print(f"\nFinal Results for {model_name}:")
            print(f"CV Accuracy: {best_histories['cv_mean']:.2f}% ± {best_histories['cv_std']:.2f}")
            print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()