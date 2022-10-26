import os.path

# sys.path.insert(0, "C:\\Users\\Carson Brown\\git\\glyphs")
#
# sys.path = [
#     "C:\\Users\\Carson Brown\\git\\glyphs",
#     "C:\\Users\\Carson Brown\\.conda\\envs\\glyphs\\python310.zip",
#     "C:\\Users\\Carson Brown\\.conda\\envs\\glyphs\\DLLs",
#     "C:\\Users\\Carson Brown\\.conda\\envs\\glyphs\\lib",
#     "C:\\Users\\Carson Brown\\.conda\\envs\\glyphs",
#     "C:\\Users\\Carson Brown\\.conda\\envs\\glyphs\\lib\\site-packages"
# ]
from src import data_extraction
from src.util import dir_util

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
GLYPH_DIR = os.path.join(DATASET_DIR, "glyphs")

TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train")
TRAIN_IMAGE_MONO_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "mono", "raw")
TRAIN_IMAGE_MONO_BINARY_DIR = os.path.join(TRAIN_IMAGE_DIR, "mono", "binarized")
TRAIN_LABEL_DIR = os.path.join(LABEL_DIR, "train")
TRAIN_LABEL_RAW_DIR = os.path.join(TRAIN_LABEL_DIR, "raw")
TRAIN_LABEL_MONO_RAW_DIR = os.path.join(TRAIN_LABEL_DIR, "mono", "raw")
TRAIN_LABEL_MONO_BINARY_DIR = os.path.join(TRAIN_LABEL_DIR, "mono", "binarized")
EVAL_IMAGE_DIR = os.path.join(IMAGE_DIR, "eval")
EVAL_IMAGE_MONO_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "mono", "raw")
EVAL_IMAGE_MONO_BINARY_DIR = os.path.join(EVAL_IMAGE_DIR, "mono", "binarized")
EVAL_LABEL_DIR = os.path.join(LABEL_DIR, "eval")
EVAL_LABEL_RAW_DIR = os.path.join(EVAL_LABEL_DIR, "raw")
EVAL_LABEL_MONO_RAW_DIR = os.path.join(EVAL_LABEL_DIR, "mono", "raw")
EVAL_LABEL_MONO_BINARY_DIR = os.path.join(EVAL_LABEL_DIR, "mono", "binarized")
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, "test")
TEST_LABEL_DIR = os.path.join(LABEL_DIR, "test")

TRAIN_IMAGE_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "raw")
EVAL_IMAGE_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "raw")
TRAIN_BINARIZED_DIR = os.path.join(TRAIN_IMAGE_DIR, "binarized")
EVAL_BINARIZED_DIR = os.path.join(EVAL_IMAGE_DIR, "binarized")

COCO_TRAINING_DIR = os.path.join("HomerCompTraining")
COCO_TESTING_DIR = os.path.join("HomerCompTesting")

RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "raw")
BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "binarized")

RAW_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "raw")
BINARIZED_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "binarized")
ARTIFICIAL_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "artificial")

# TRAIN_RESIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "raw")
# TRAIN_RESIZED_BINARIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "binarized")
# TRAIN_OCULAR_TRAIN_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "ocular_training")
# TRAIN_OUTPUT_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "output")
# RAW_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "raw")
# BINARIZED_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "binarized")

# binarize.cnn(INPUT_DIR, CONFIDENCE_DIR, threshold=None)

data_extraction.generate_yolo_labels(COCO_TRAINING_DIR, TRAIN_LABEL_MONO_BINARY_DIR, mono_class=True)
dir_util.split_eval_data(TRAIN_IMAGE_MONO_BINARY_DIR, TRAIN_LABEL_MONO_BINARY_DIR, EVAL_IMAGE_MONO_BINARY_DIR, EVAL_LABEL_MONO_BINARY_DIR)


# data_extraction.extract_glyphs(COCO_TRAINING_DIR,1)
#                                INPUT_DIR,
#                                RAW_TEMPLATE_GLYPHS_DIR,
#                                quality_filter=["bt1"],
#                                glyphs_per_footmark_type_limit=1)
# data_extraction.extract_glyphs(COCO_TRAINING_DIR,
#                                BINARIZED_DIR,
#                                BINARIZED_TEMPLATE_GLYPHS_DIR,
#                                quality_filter=["bt1"],
#                                glyphs_per_footmark_type_limit=1)

# templates_vector, template_class = classify.alex_init(ARTIFICIAL_TEMPLATE_GLYPHS_DIR, overwrite=True)
# all_vectors, all_classes = classify.alex_init(BINARIZED_GLYPHS_DIR)
# classify.alex_knn(templates_vector, template_class, all_vectors, all_classes)

# yolo.train_yolo(os.path.join(DATASET_DIR, "yolov5_raw.yml"), 1)
