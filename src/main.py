from binarization import binarize
from line_recognition import find_lines

# TRAIN
# INPUT_DIR = "dataset/train/raw"
# BINARIZED_DIR = "dataset/train/binarized"

# TEST
INPUT_DIR = "dataset/test/raw"
FILTERED_DIR = "dataset/test/filtered"
BINARIZED_DIR = "dataset/test/binarized"
LINES_DIR = "dataset/test/lines"

binarize.gabor(INPUT_DIR, FILTERED_DIR)
binarize.cnn(FILTERED_DIR, BINARIZED_DIR)
# find_lines.tesseract(BINARIZED_DIR, LINES_DIR)


