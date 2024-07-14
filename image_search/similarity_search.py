import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import faiss

class SimilaritySearch:
    def __init__(self, features_list, filenames):
        self.features_list = features_list
        self.filenames = filenames
        self.index = faiss.IndexFlatL2(features_list.shape[1])
        self.index.add(features_list)

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

    def find_similar_images_faiss(self, query_features, top_n=10):
        query_features = query_features.reshape(1, -1)
        distances, indices = self.index.search(query_features, top_n)
        return indices[0], distances[0]
