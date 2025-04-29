import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('your_model.h5')  # If you saved it

# Load and preprocess your own image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')        # Convert to grayscale
    img = img.resize((28, 28))                       # Resize to 28x28
    img_array = np.array(img)
    img_array = 255 - img_array                      # Invert colors (white background)
    img_array = img_array / 255.0                    # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)       # Reshape for model
    return img_array

# Predict
image = preprocess_image('digit.png')
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)

plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.show()
