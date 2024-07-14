import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract_features(self, image):
        image = cv2.resize(image, (224, 224))  # Resize image to match model's expected sizing
        image = img_to_array(image)  # Convert image to numpy array
        image = np.expand_dims(image, axis=0)  # Add one more dimension
        image = preprocess_input(image)  # Preprocess the image
        features = self.model.predict(image)  # Extract features
        return features
