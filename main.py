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
    
    print(X.shape, y.shape)
    
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
    return y
    
X, y = load_dataset('data/chest_xray/train')    
W1, b1, W2, b2 = initialise_parameters(4096, 64, 1)
y_hat = forward_pass(X, W1, b1, W2, b2)
print(y_hat.shape)
print(y_hat[:5])

