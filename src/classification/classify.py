import json
import os.path
from math import floor

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from src.classification.alexnet import alexnet
from src.util.dir_util import get_input_images

ALEXNET_KNOWN_VECTOR_PATH = "alexnet_vectors.json"


def alex_init(categories_path, *, overwrite=False):
    category_dict = get_input_images(categories_path, by_dir=True)
    tensors = {}
    full_tensor_path = os.path.join(categories_path, ALEXNET_KNOWN_VECTOR_PATH)
    if overwrite or not os.path.exists(full_tensor_path):
        for category, img_paths in category_dict.items():
            print(f"Classifying {category}...")
            tensors[category] = [alexnet.classify(img_path) for img_path in img_paths]
        print("Writing AlexNet Data To Disk")
        with open(full_tensor_path, mode="w") as f:
            f.write(json.dumps(tensors, indent=4))
    else:
        print("Reading AlexNet Data From Disk")
        with open(full_tensor_path, mode="r") as f:
            tensors = json.loads(f.read())
    return tensors


def alex_cluster(known_tensors):
    """
    Author: https://builtin.com/data-science/data-clustering-python
    :param known_tensors: the dict with the key being the category and the
    value being a list of all the tensors in that category
    :return:
    """
    print("Training AlexNet GMM")

    n_clusters = len(known_tensors)
    train_tensors, _, test_tensors, test_truth = split_data(known_tensors)

    gmm_model = GaussianMixture(n_components=n_clusters)
    gmm_model.fit(train_tensors)
    cluster_labels = gmm_model.predict(test_tensors)
    print(mutual_info_score(test_truth, cluster_labels))
    print(adjusted_mutual_info_score(test_truth, cluster_labels))


def alex_forest(known_tensors):
    print("Training AlexNet Random Forest")
    train_tensors, train_truth, test_tensors, test_truth = split_data(known_tensors)
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(train_tensors, train_truth)

    predictions = clf.predict(test_tensors)
    print(classification_report(test_truth, predictions))


def alex_knn(known_vectors):
    print("Training AlexNet KNN")
    train_tensors, train_truth, test_tensors, test_truth = split_data(known_vectors)
    clf = KNeighborsClassifier(n_neighbors=50, weights="distance")
    clf.fit(train_tensors, train_truth)

    predictions = clf.predict(test_tensors)
    print(classification_report(test_truth, predictions))


def split_data(known_tensors, train_percent=0.7):
    train_tensors, train_truth, test_tensors, test_truth = [], [], [], []
    for i, t in enumerate(known_tensors.values()):
        train_t = t[floor(len(t) * train_percent):]
        test_t = t[:floor(len(t) * train_percent)]
        train_tensors += train_t
        train_truth += [i] * len(train_t)
        test_tensors += test_t
        test_truth += [i] * len(test_t)

    return train_tensors, train_truth, test_tensors, test_truth


def alex_classify(image_path):
    alexnet.classify(image_path)
