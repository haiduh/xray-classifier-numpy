from PIL import Image
import numpy as np
import os
from pathlib import Path

def image_processing(path: str):
    image = Image.open(path).convert('L')   
    resized_image = image.resize((64,64))
    image_array = np.array(resized_image)
    normalised_image = image_array/255
    flattened_image = normalised_image.flatten()
    return flattened_image

def load_dataset(path: str):
    base = Path(path)
    x = []
    y = []
    
    normal_files = list((base / 'NORMAL').rglob('*'))
    pneumonia_files = list((base / 'PNEUMONIA').rglob('*'))
    
    # Filter out directories just in case, keeping only files
    normal_files = [f for f in normal_files if f.is_file()]
    pneumonia_files = [f for f in pneumonia_files if f.is_file()]
    
    for f in normal_files:
        if f.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        flattened_image = image_processing(f)
        x.append(flattened_image)
        y.append(0)      
    
    for f in pneumonia_files:
        if f.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        flattened_image = image_processing(f)
        x.append(flattened_image)
        y.append(1)
        
    X = np.array(x)
    y = np.array(y)
    
    return X, y

def initialise_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden_size,))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((output_size,))
    return W1, b1, W2, b2

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forward_pass(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    y = sigmoid(z2)
    return y, z1, a1

def loss_function(y_true, y_hat):
    y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8) # clip because np.log(0) is -inf, break training if the network ever predicts exactly 0 or 1
    bce = -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))   # Binary cross entropy function translated to Python
    return bce

def backward_pass(X, y, y_hat, W2, Z1, A1): # d means derivative
    N = X.shape[0]
    dZ2 = y_hat - y
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0).astype(float)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1)
    
    return dW1/N, db1/N, dW2/N, db2/N

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    
    return W1, b1, W2, b2

def train(X, y, learning_rate, epochs):
    W1, b1, W2, b2 = initialise_parameters(4096, 64, 1)
    
    for epoch in range(epochs):
        y_hat, Z1, A1 = forward_pass(X, W1, b1, W2, b2)
        loss = loss_function(y, y_hat)
        dW1, db1, dW2, db2 =backward_pass(X, y, y_hat, W2, Z1, A1)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    return W1, b1, W2, b2 
        
def accuracy(y, y_hat):
    y_pred = (y_hat >= 0.5).astype(int) # can use for loop but y_pred was a plain python list so the comparison with y (a 2D NumPy array) it fumbled
    accuracy = np.mean(y_pred == y)
    return accuracy

    
# Train set
X, y = load_dataset('data/chest_xray/train')
y = y.reshape(-1, 1)  
W1, b1, W2, b2 = train(X, y, 0.001, 500)
y_hat, _, _ = forward_pass(X, W1, b1, W2, b2) # need to unpack every element because Z1 and A1 were added 
print(f"Train accuracy: {accuracy(y, y_hat):.4f}")

# Test set
X_test, y_test = load_dataset('data/chest_xray/test')
y_test = y_test.reshape(-1, 1)
y_hat_test, _, _ = forward_pass(X_test, W1, b1, W2, b2)
print(f"Test accuracy: {accuracy(y_test, y_hat_test):.4f}")

