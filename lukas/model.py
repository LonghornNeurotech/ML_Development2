# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    """
    A simple neural network with one hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example instantiation for testing purposes
    model = MyModel(input_dim=10, hidden_dim=20, output_dim=2)
    print(model)
