from binarization import binarize

# INPUT_DIR = "dataset/train/raw"
BINARIZED_DIR = "dataset/train/binarized"

INPUT_DIR = "dataset/test/raw"
BINARIZED_DIR = "dataset/test/binarized"

# binarize.cnn(INPUT_DIR, BINARIZED_DIR)
binarize.sensitive_threshold(INPUT_DIR, BINARIZED_DIR)
