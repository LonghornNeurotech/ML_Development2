# EEG RBF Network Project

This project implements a Radial Basis Function (RBF) neural network for classifying EEG signals. The project is structured into multiple modules that handle data preprocessing, model construction, training, and evaluation. This README provides a high-level overview of the key steps and components of the code.

---

## Project Overview

The project is divided into four main files:

- **pre_process.py**  
  Handles data extraction, signal filtering, segmentation, channel reordering, and creation of PyTorch datasets and DataLoaders.

- **model.py**  
  Defines the RBF network architecture, including a custom fixed-gamma RBF layer, and includes a helper function to initialize the RBF centers using KMeans clustering.

- **train.py**  
  Loads the preprocessed data, constructs and trains the RBF network using the training dataset, and saves the trained model to disk.

- **evaluate.py**  
  Loads the trained model and evaluates its performance on the validation and test datasets.

---

## Preprocessing Steps

1. **Data Extraction:**  
   The EEG data is provided as a zipped archive (`LHNT_EEG.zip`). The `pre_process.py` script unzips the archive and searches for EEG data stored as pickle files within subfolders.

2. **Loading and Labeling:**  
   Each pickle file is loaded and the EEG signal is extracted. Labels are assigned based on the filename (e.g., filenames containing "right" are labeled as 1; otherwise, 0).

3. **Signal Processing:**  
   - **Channel Reordering:**  
     The channels of the EEG signals are rearranged into a specified order.
   - **Bandpass Filtering:**  
     A Butterworth bandpass filter (default: 5–40 Hz) is applied to remove noise and unwanted frequency components.
   - **Normalization:**  
     The filtered signals are normalized channel-wise.
   - **Segmentation:**  
     Each signal is segmented into overlapping windows (default window size: 1.5 seconds with a 0.0175-second shift) to generate multiple training examples per original recording.
   - **Column Removal:**  
     Specific channels (columns) are removed from the segmented data to reduce dimensionality or remove noisy channels.

4. **Dataset Preparation:**  
   The processed segments and their one-hot encoded labels are converted to PyTorch tensors and organized into TensorDatasets. The data is split into training, validation, and test sets with corresponding DataLoaders for batch processing.

---

## Model Architecture

1. **Fixed Gamma RBF Layer:**  
   - A custom PyTorch layer (`FixedGammaRBFLayer`) computes radial basis function activations using a fixed gamma parameter.
   - It contains trainable parameters for the centers (initialized using Xavier initialization) and scaling weights.

2. **RBF Network:**  
   - The network (`RBFNetwork`) begins by flattening the input EEG segments.
   - The flattened input is passed through the RBF layer, transforming it into a new feature space.
   - A two-layer fully connected classifier (with ReLU activation and dropout) processes the RBF features to output class scores.

3. **RBF Center Initialization:**  
   - A helper function uses KMeans clustering on a subset of the training data to initialize the centers of the RBF layer, which can help improve training convergence.

---

## Training Procedure

- **Training Script (train.py):**  
  - Loads the preprocessed DataLoaders.
  - Constructs the RBF network using parameters such as the number of RBF units, gamma, hidden layer size, and dropout probability.
  - Optionally initializes the RBF centers using KMeans clustering on a subset of training data.
  - Uses the Adam optimizer with a learning rate of 0.001 and weight decay for regularization.
  - Employs cross-entropy loss for classification.
  - Trains the network for a specified number of epochs (default: 20), printing the average training loss per epoch.
  - Saves the trained model weights to `rbf_model.pth`.

---

## Evaluation Criteria

- **Evaluation Script (evaluate.py):**  
  - Loads the saved model weights.
  - Evaluates the model on both the validation and test datasets.
  - Computes the accuracy by comparing the predicted labels (obtained by taking the class with the highest score) with the true labels.
  - Prints out the accuracy for each dataset, providing a quantitative measure of model performance.

---

## Dependencies

To run this project, you will need:
- Python 3.x
- PyTorch
- NumPy
- SciPy
- scikit-learn
- matplotlib

Install the dependencies using pip:

```bash
pip install torch numpy scipy scikit-learn matplotlib
