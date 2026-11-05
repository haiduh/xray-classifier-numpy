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
    
print(load_dataset('data/chest_xray/train'))

