from sklearn.neighbors import KNeighborsClassifier


def init_knn(template_vectors, template_labels, n=5):
    print(f"Training KNN with N={n}")
    knn = KNeighborsClassifier(n_neighbors=n, metric='cosine', weights="distance")
    knn.fit(template_vectors, template_labels)
    return knn


def vector_knn(knn, vectors):
    return knn.predict_proba(vectors)
