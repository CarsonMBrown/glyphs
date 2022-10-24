import json
import os.path
import os.path
import pickle
from math import floor

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, classification_report, top_k_accuracy_score, \
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from src.classification.alexnet import alexnet
from src.util.dir_util import get_input_images
from src.util.glyph_util import get_classes_as_glyphs

ALEXNET_KNOWN_VECTOR_PATH = "alexnet_vectors.pickle"
glyph_classes = get_classes_as_glyphs()


def alex_init(categories_path, *, write=True, overwrite=False):
    category_dict = get_input_images(categories_path, by_dir=True)
    vectors = {}
    full_vector_path = os.path.join(categories_path, ALEXNET_KNOWN_VECTOR_PATH)
    if overwrite or not os.path.exists(full_vector_path):
        for category, img_paths in category_dict.items():
            print(f"Classifying {category}...")
            vectors[category] = [alexnet.classify(img_path) for img_path in img_paths]
        print("Writing AlexNet Data To Disk")
        if write:
            with open(full_vector_path, mode="wb") as f:
                pickle.dump(vectors, f)
    else:
        print("Reading AlexNet Data From Disk")
        with open(full_vector_path, mode="rb") as f:
            vectors = pickle.load(f)
    return vector_class_pair(vectors)


def alex_cluster(n_clusters, train_vectors, test_vectors, test_truth):
    print("Training AlexNet GMM")
    gmm_model = GaussianMixture(n_components=n_clusters)
    gmm_model.fit(train_vectors)
    cluster_labels = gmm_model.predict(test_vectors)
    print(mutual_info_score(test_truth, cluster_labels))
    print(adjusted_mutual_info_score(test_truth, cluster_labels))


def alex_forest(train_vectors, train_truth, test_vectors, test_truth):
    print("Training AlexNet Random Forest")
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(train_vectors, train_truth)

    predictions = clf.predict(test_vectors)
    probs = clf.predict_proba(test_vectors)
    print(classification_report(test_truth, predictions))
    print(top_k_accuracy_score(test_truth, probs, k=2))
    print(top_k_accuracy_score(test_truth, probs, k=3))


def alex_knn(train_vectors, train_truth, test_vectors, test_truth):
    print("Training AlexNet KNN")
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(train_vectors, train_truth)

    predictions = clf.predict(test_vectors)
    probs = clf.predict_proba(test_vectors)
    print(classification_report(test_truth, predictions))
    print(top_k_accuracy_score(test_truth, probs))
    print(top_k_accuracy_score(test_truth, probs, k=3))

    cm = confusion_matrix(test_truth, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=glyph_classes)
    disp.plot()
    plt.show()


def split_data(known_vectors, train_percent=0.7):
    train_vectors, train_truth, test_vectors, test_truth = [], [], [], []
    for i, t in enumerate(known_vectors.values()):
        train_t = t[:floor(len(t) * train_percent)]
        train_vectors += train_t
        train_truth += [i] * len(train_t)
        test_t = t[floor(len(t) * train_percent):]
        test_vectors += test_t
        test_truth += [i] * len(test_t)

    return train_vectors, train_truth, test_vectors, test_truth


def vector_class_pair(known_vectors):
    vectors, truth = [], []
    for i, t in enumerate(known_vectors.values()):
        vectors += t
        truth += [list(known_vectors.keys())[i]] * len(t)
    return vectors, truth


def alex_classify(image_path):
    alexnet.classify(image_path)
