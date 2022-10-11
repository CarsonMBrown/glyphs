import os.path

from src.preprocessing import preprocess

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

# BINARIZE AND THEN GET GLYPHS
# binarize.cnn(INPUT_DIR, BINARIZED_DIR)
# data_extraction.extract_glyphs("HomerCompTraining", BINARIZED_DIR, BINARIZED_GLYPHS_DIR)

# SCALE BINARY THEN OCULAR
preprocess.scale(INPUT_DIR, RESIZED_DIR, 1024, 1024)
preprocess.scale(BINARIZED_DIR, RESIZED_BINARIZED_DIR, 1024, 1024)
# ocular.run_ocular(RESIZED_BINARIZED_DIR, OUTPUT_DIR, overwrite=False)

# # SCALE THEN OCULAR
# preprocess.scale(INPUT_DIR, PROCESSED_DIR, 1000, 1000)
# ocular.init_ocular(PROCESSED_DIR, OUTPUT_DIR, overwrite=False)
# ocular.transcribe(PROCESSED_DIR, OUTPUT_DIR)
