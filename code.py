
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Paths to training and testing directories
train_dir = "D:\CVIP_2\Project\Leaf-Disease-Detection-System\train"
test_dir = "D:\CVIP_2\Project\Leaf-Disease-Detection-System\test"

# Image preprocessing parameters
img_size = (224, 224)  # Resize images to 224x224
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,   # Normalize pixel values to range [0, 1]
    rotation_range=20,   # Rotate images
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,     # Shear transformation
    zoom_range=0.2,      # Zoom in/out
    horizontal_flip=True # Flip images horizontally
)

# Only rescale images for testing (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load and preprocess training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=True
)

# Load and preprocess testing data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Save preprocessed datasets as numpy arrays
def save_dataset_as_numpy(data_generator, output_dir, dataset_type):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    labels = []
    for batch_idx in range(len(data_generator)):
        batch_images, batch_labels = data_generator[batch_idx]
        images.extend(batch_images)
        labels.extend(batch_labels)
    
    images = np.array(images)
    labels = np.array(labels)
    
    np.save(os.path.join(output_dir, f"{dataset_type}_images.npy"), images)
    np.save(os.path.join(output_dir, f"{dataset_type}_labels.npy"), labels)
    print(f"Saved {dataset_type} data: {images.shape}, {labels.shape}")

# Save training and testing datasets
save_dataset_as_numpy(train_data, "preprocessed_data", "training")
save_dataset_as_numpy(test_data, "preprocessed_data", "testing")
