# preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Preprocesses the raw data:
    - Separates features and labels.
    - Scales feature values.
    """
    # Assuming the last column is the label
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def get_train_test_data(filepath, test_size=0.2, random_state=42):
    """
    Loads and preprocesses the data, then splits it into training and testing sets.
    """
    data = load_data(filepath)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage: adjust 'data.csv' to your actual data file path
    X_train, X_test, y_train, y_test = get_train_test_data("data.csv")
    print("Data loaded and preprocessed successfully.")
