# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from model import MyModel
from preprocess import get_train_test_data

def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    """
    Trains the model using Adam optimizer and CrossEntropy loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # Convert training data to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

if __name__ == '__main__':
    # Adjust 'data.csv' to your actual data file path.
    X_train, X_test, y_train, y_test = get_train_test_data("data.csv")
    input_dim = X_train.shape[1]
    hidden_dim = 50
    output_dim = len(set(y_train))
    
    model = MyModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    trained_model = train_model(model, X_train, y_train, epochs=20, lr=0.001)
