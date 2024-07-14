import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from image_search.feature_extraction import FeatureExtractor
from image_search.similarity_search import SimilaritySearch
from image_search.utils import load_images_from_folder

# Function to extract and save features
def extract_and_save_features(images, filenames, extractor, features_file, filenames_file):
    features_list = [extractor.extract_features(img) for img in images]
    features_list = np.array(features_list).reshape(len(features_list), -1)
    np.save(features_file, features_list)
    np.save(filenames_file, filenames)
    return features_list

# Function to load dataset and extract features
def load_dataset(dataset_dir):
    images, filenames = load_images_from_folder(dataset_dir)
    extractor = FeatureExtractor()

    features_file = 'features_list.npy'
    filenames_file = 'filenames.npy'

    if os.path.exists(features_file) and os.path.exists(filenames_file):
        print("Loading features from file...")
        features_list = np.load(features_file)
        filenames = np.load(filenames_file)
    else:
        print("Extracting features and saving to file...")
        features_list = extract_and_save_features(images, filenames, extractor, features_file, filenames_file)

    similarity_search = SimilaritySearch(features_list, filenames)
    return similarity_search, filenames

# Function to process an uploaded image
def process_image(extractor, image_path, similarity_search, filenames):
    try:
        # Load query image
        query_image = cv2.imread(image_path)
        if query_image is None:
            raise ValueError("Failed to load image. Please provide a valid image file path.")

        # Extract features from query image
        #extractor = FeatureExtractor()
        query_features = extractor.extract_features(query_image)

        # Find similar images using cosine similarity
        indices_cosine, _ = similarity_search.find_similar_images_cosine(query_features)
        similar_images_cosine = [filenames[i] for i in indices_cosine]

        # Find similar images using Euclidean distance
        # indices_euclidean, _ = similarity_search.find_similar_images_euclidean(query_features)
        # similar_images_euclidean = [filenames[i] for i in indices_euclidean]

        # Initialize Matplotlib figure for input and similar images
        plt.figure(figsize=(18, 8))
        
        # Plot input image
        plt.subplot(3, 5, 1)
        plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')

        # Plot top 10 similar images using Cosine Similarity
        for i, filename in enumerate(similar_images_cosine[:10]):
            image_path = os.path.join('dataset', filename)
            image = cv2.imread(image_path)
            plt.subplot(3, 5, i+6)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Output {i+1}")
            plt.axis('off')

        # Display top 10 similar images using Euclidean Distance
        # plt.figure(figsize=(18, 6))
        # plt.suptitle('Top 10 similar images using Euclidean Distance', fontsize=16)
        # for i, filename in enumerate(similar_images_euclidean[:10]):
        #     image_path = os.path.join('dataset', filename)
        #     image = cv2.imread(image_path)
        #     plt.subplot(2, 5, i+1)
        #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     plt.title(filename)
        #     plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        return True

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")

# Main function
def main(dataset_dir):
    similarity_search, filenames = load_dataset(dataset_dir)
    extractor = FeatureExtractor()
    while True:
        image_path = input("Please Enter Image name = ")
        image_path = dataset_dir+image_path
        process_image(extractor, image_path, similarity_search, filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Search for Garments")
    #parser.add_argument("--image_path", type=str, required=True, help="Path to the query image")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to the dataset directory")
    args = parser.parse_args()

    main(args.dataset_dir)
