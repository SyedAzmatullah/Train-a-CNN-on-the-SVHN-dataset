# Train-a-CNN-on-the-SVHN-dataset
# Install necessary libraries
!pip install mat73

import mat73
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define dataset paths
train_dataset_path = '/content/drive/MyDrive/Colab/train_32x32.mat'
test_dataset_path = '/content/drive/MyDrive/Colab/test_32x32.mat'

# Load data function
def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)  # Since the original dataset isn't using mat73 but loadmat
    return data['X'], data['y']

# Load training and test data
X_train, y_train = load_data(train_dataset_path)
X_test, y_test = load_data(test_dataset_path)

# Inspect the shape
print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)

# Transpose the dataset: (width, height, channels, samples) -> (samples, width, height, channels)
X_train = np.transpose(X_train, (3, 0, 1, 2))
X_test = np.transpose(X_test, (3, 0, 1, 2))

print("Reshaped Training Set", X_train.shape, y_train.shape)
print("Reshaped Test Set", X_test.shape, y_test.shape)

# Normalize the image data (from 0-255 to 0-1)
X_train = X_train.astype('float32')/ 255.0
X_test = X_test.astype('float32')/ 255.0

# Handle label 10 (which represents '0') as 0 for classification
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training Set", X_train.shape, y_train.shape)
print("Validation Set", X_val.shape, y_val.shape)
print("Test Set", X_test.shape, y_test.shape)

### Building the CNN Model
def build_model():
    model = Sequential()

    # First convolution layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolution layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolution layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer (10 digits, softmax activation for classification)
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = build_model()

# Print model summary
model.summary()

### Training the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

### Evaluating the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

### Plotting Training History
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# Function to display pixel values for an image
def show_image_pixel_values(index):
    """Displays the pixel values of an image from the test set."""

    # Get the image
    image = X_test[index]

    # Convert to grayscale to simplify pixel display (optional)
    grayscale_image = np.mean(image, axis=2)  # Averaging over RGB channels

    # Display the pixel values as a heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(grayscale_image, annot=True, fmt=".1f", cmap="viridis", cbar=False)
    plt.title(f"Pixel Values for Image at Index {index}")
    plt.axis('off')  # Hide axes
    plt.show()   # prompt: write code to make user interface so that user give imput image from pc

from google.colab import files
from PIL import Image
import io
import numpy as np

def predict_image():
    uploaded = files.upload()
    for fn in uploaded.keys():
        image = Image.open(io.BytesIO(uploaded[fn]))
        image = image.resize((32, 32))  # Resize to match the model's input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        print(f"Predicted class: {predicted_class}")

# Create a button to trigger image upload and prediction
import ipywidgets as widgets
button = widgets.Button(description="Upload Image and Predict")
button.on_click(lambda b: predict_image())
display(button
