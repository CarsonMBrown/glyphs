from bounding import bound

INPUT_DIR = "dataset/test/raw"
FILTERED_DIR = "dataset/test/filtered"
TEMP_DIR = "dataset/test/temp"
BINARIZED_DIR = "dataset/test/binarized"
BOUND_DIR = "dataset/test/bound"
MASK_DIR = "dataset/test/masks"
LINES_DIR = "dataset/test/lines"

bound.connected_component(BINARIZED_DIR, BOUND_DIR)
