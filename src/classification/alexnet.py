import os.path
import pickle
from math import floor

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, classification_report, \
    top_k_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from torchvision.models import AlexNet_Weights

from src.util.dir_util import get_input_images
from src.util.glyph_util import get_classes_as_glyphs

KNOWN_VECTOR_FILE = "alexnet_vectors.pickle"
GLYPH_CLASSES = get_classes_as_glyphs()


def classify(image):
    """

    Author: https://pytorch.org/hub/pytorch_vision_alexnet/
    :param image:
    :return:
    """
    input_image = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        alex_net.to('cuda')

    with torch.no_grad():
        output = alex_net(input_batch)

    return output.tolist()[0]


def alex_init(categories_path, *, write=True, overwrite=False):
    category_dict = get_input_images(categories_path, by_dir=True)
    vectors = {}
    full_vector_path = os.path.join(categories_path, KNOWN_VECTOR_FILE)
    if overwrite or not os.path.exists(full_vector_path):
        for category, img_paths in category_dict.items():
            print(f"Classifying {category}...")
            vectors[category] = [classify(img_path) for img_path in img_paths]
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


def alex_distance(v1, v2):
    v1_n = v1 / np.linalg.norm(v1)
    v2_n = v2 / np.linalg.norm(v2)
    return distance.euclidean(v1_n, v2_n)


def alex_knn(train_vectors, train_truth, test_vectors, test_truth):
    print("Training AlexNet KNN")
    clf = KNeighborsClassifier(n_neighbors=5, metric=alex_distance, weights="distance")
    clf.fit(train_vectors, train_truth)

    print("Evaluating AlexNet KNN")
    predictions = clf.predict(test_vectors)
    probs = clf.predict_proba(test_vectors)
    print(classification_report(test_truth, predictions))
    print(top_k_accuracy_score(test_truth, probs))
    print(top_k_accuracy_score(test_truth, probs, k=3))

    cm = confusion_matrix(test_truth, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GLYPH_CLASSES)
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


# instantiate the model
alex_net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
alex_net.classifier = nn.Sequential(*[alex_net.classifier[i] for i in range(1)])
alex_net.eval()
