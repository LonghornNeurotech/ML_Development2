# evaluate.py

import torch
from model import MyModel
from preprocess import get_train_test_data
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and prints the accuracy.
    """
    model.eval()
    inputs = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # Obtain predicted classes
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predicted.numpy())
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':
    # Adjust 'data.csv' to your actual data file path.
    X_train, X_test, y_train, y_test = get_train_test_data("data.csv")
    input_dim = X_train.shape[1]
    hidden_dim = 50
    output_dim = len(set(y_train))
    
    # Initialize the model (in practice, you would load your trained model)
    model = MyModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    evaluate_model(model, X_test, y_test)
