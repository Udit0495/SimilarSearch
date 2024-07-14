import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class SimilaritySearch:
    def __init__(self, features_list, filenames):
        self.features_list = features_list
        self.filenames = filenames

    def find_similar_images_cosine(self, query_features, top_n=10):
        similarities = cosine_similarity(query_features, self.features_list)
        similarities = similarities.flatten()
        indices = np.argsort(similarities)[::-1][1:top_n+1]
        return indices, similarities[indices]

    def find_similar_images_euclidean(self, query_features, top_n=10):
        distances = euclidean_distances(query_features, self.features_list)
        distances = distances.flatten()
        indices = np.argsort(distances)[1:top_n+1]
        return indices, distances[indices]
