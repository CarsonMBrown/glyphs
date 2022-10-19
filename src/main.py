import os.path

DATASET_DIR = "dataset"
INNER_DIR = "train"
GLYPH_DIR = os.path.join(DATASET_DIR, "glyphs")

INPUT_DIR = os.path.join(DATASET_DIR, INNER_DIR, "raw")
BINARIZED_DIR = os.path.join(DATASET_DIR, INNER_DIR, "binarized")
RESIZED_DIR = os.path.join(DATASET_DIR, INNER_DIR, "resized", "raw")
RESIZED_BINARIZED_DIR = os.path.join(DATASET_DIR, INNER_DIR, "resized", "binarized")
OCULAR_TRAIN_DIR = os.path.join(DATASET_DIR, INNER_DIR, "ocular_training")
OUTPUT_DIR = os.path.join(DATASET_DIR, INNER_DIR, "output")

RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "raw")
BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "binarized")
