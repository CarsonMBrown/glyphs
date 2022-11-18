import os.path
from functools import partial

import cv2
from numpy import arange

from src.binarization import binarize
from src.bounding.yolo import yolo
from src.classification.cnn_learning.resnext_lstm import ResNext101LSTM, ResNextLongLSTM
from src.classification.markov import markov
from src.classification.vector_learning import nn_factory
from src.data_extraction import extract_cropped_glyphs
from src.evaluation.bbox_eval import remove_bbox_outliers, get_bbox_outliers
from src.line_recognition.bbox_connection import link_bboxes, remove_bbox_intersections
from src.util.glyph_util import get_classes_as_glyphs
from src.util.img_util import plot_bboxes, plot_lines
from src.util.line_util import get_line_centers
from src.util.torch_dataloader import ImageLoader

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
GLYPH_DIR = os.path.join(DATASET_DIR, "glyphs")

ALL_IMAGE_DIR = os.path.join(IMAGE_DIR, "all")
ALL_RAW_DIR = os.path.join(ALL_IMAGE_DIR, "raw")
ALL_BINARIZED_DIR = os.path.join(ALL_IMAGE_DIR, "binarized")

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

TRAIN_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "raw")
EVAL_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "raw")
TRAIN_BINARIZED_DIR = os.path.join(TRAIN_IMAGE_DIR, "binarized")
EVAL_BINARIZED_DIR = os.path.join(EVAL_IMAGE_DIR, "binarized")

COCO_TRAINING_DIR = os.path.join("HomerCompTraining")
COCO_TESTING_DIR = os.path.join("HomerCompTesting")

RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "raw")
TRAIN_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "raw")
TRAIN_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "cropped_raw")
TRAIN_RAW_QUALITY_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "raw_high_quality")
TRAIN_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "binarized")
EVAL_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "raw")
EVAL_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "cropped_raw")
BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "binarized")
EVAL_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "binarized")

RAW_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "raw")
BINARIZED_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "binarized")
ARTIFICIAL_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "artificial")

lang_file = os.path.join(DATASET_DIR, "perseus_25000.txt"), os.path.join(DATASET_DIR, "perseus_5000.txt")
quick_lang_file = os.path.join(DATASET_DIR, "perseus_5000.txt"), os.path.join(DATASET_DIR, "perseus_2000.txt")
meta_data = os.path.join(GLYPH_DIR, "meta.csv")


# TRAIN_RESIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "raw")
# TRAIN_RESIZED_BINARIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "binarized")
# TRAIN_OCULAR_TRAIN_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "ocular_training")
# TRAIN_OUTPUT_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "output")
# RAW_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "raw")
# BINARIZED_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "binarized")


def train_model():
    nn_factory.train_model(quick_lang_file, meta_data,
                           TRAIN_RAW_GLYPHS_DIR, EVAL_RAW_GLYPHS_DIR,
                           ResNext101LSTM,
                           epochs=200, batch_size=8, num_workers=0, resume=True, start_epoch=55, loader=ImageLoader,
                           transforms=[ResNext101LSTM.transform_train, ResNext101LSTM.transform_classify])


def eval_model():
    eval_dataset, eval_dataloader = nn_factory.generate_dataloader(os.path.join(DATASET_DIR, "perseus_25000.txt"),
                                                                   meta_data,
                                                                   EVAL_RAW_GLYPHS_DIR, batch_size=8,
                                                                   loader=ImageLoader,
                                                                   transform=ResNext101LSTM.transform_classify)
    print("Epoch, Precision, Recall, FScore")
    for epoch in range(0, 73):
        model, _ = nn_factory.load_model(
            ResNextLongLSTM, load_epoch=epoch, dataset=eval_dataset, resume=False)
        avg_precision, avg_recall, avg_fscore = nn_factory.eval_model(model, eval_dataloader, average="weighted",
                                                                      seed=0)
        print(f"{epoch}, {avg_precision}, {avg_recall}, {avg_fscore}")


def deep_eval_model():
    eval_dataset, eval_dataloader = nn_factory.generate_dataloader(os.path.join(DATASET_DIR, "perseus_25000.txt"),
                                                                   meta_data,
                                                                   EVAL_RAW_GLYPHS_DIR, batch_size=8,
                                                                   loader=ImageLoader,
                                                                   transform=ResNext101LSTM.transform_classify)

    log_markov_chain = markov.init_markov_chain(os.path.join(DATASET_DIR, "perseus.txt"),
                                                get_classes_as_glyphs(),
                                                cache_path=os.path.join("dataset", "perseus_log.markov"),
                                                log=True,
                                                overwrite=False)
    markov_chain = markov.init_markov_chain(os.path.join(DATASET_DIR, "perseus.txt"),
                                            get_classes_as_glyphs(),
                                            cache_path=os.path.join("dataset", "perseus.markov"),
                                            overwrite=False)

    model, _ = nn_factory.load_model(ResNextLongLSTM, load_epoch=24, dataset=eval_dataset, resume=False)

    cm, (p, r, fs, _) = nn_factory.model_confusion_matrix(model, eval_dataloader,
                                                          display_cm=False, seed=0, top_k=1,
                                                          prediction_modifier=partial(
                                                              markov.top_n_markov_optimization,
                                                              log_markov_chain,
                                                              n=1,
                                                          ))
    print(p, r, fs)

    for u in arange(0, 1.01, 0.01):
        u = round(u, 2)
        cm, (p, r, fs, _) = nn_factory.model_confusion_matrix(model, eval_dataloader,
                                                              display_cm=False, seed=0, top_k=1,
                                                              prediction_modifier=partial(
                                                                  markov.top_n_markov_optimization,
                                                                  log_markov_chain,
                                                                  n=2,
                                                                  uncertainty_threshold=u
                                                              ))
        print(u, p, r, fs)


def generate_line_images():
    input_path = "dataset/display/line_generation/"
    export_path = "output_data/line_generation/"
    img_names = [
        # "P_Hamb_graec_665",
        # "PSI_XIV_1377r",
        "G_02317_26742_Pap",
        # "G_26734_c"
    ]
    for img_name in img_names:
        display_lines(cv2.imread(input_path + img_name + ".jpg"),
                      save_path=export_path + img_name + ".png")


def display_lines(brg_img, *, save_path=None):
    rgb_img = cv2.cvtColor(brg_img, cv2.COLOR_BGR2RGB)
    bboxes = yolo.sliding_glyph_window(rgb_img)

    valid_bboxes, outlier_boxes = remove_bbox_outliers(bboxes), get_bbox_outliers(bboxes)

    lines, _ = link_bboxes(valid_bboxes)
    lines = remove_bbox_intersections(lines)

    overlay = brg_img.copy()

    valid_bboxes = [bbox for line in lines for bbox in line]

    plot_bboxes(brg_img, valid_bboxes, color=(0, 0, 0), wait=None)

    # plot_bboxes(overlay, over_size_bboxes, color=(0, 0, 255), wait=None)
    # plot_bboxes(overlay, duplicate_bboxes, color=(0, 165, 255), wait=None)

    alpha = 1
    img = cv2.addWeighted(overlay, 1 - alpha, brg_img, alpha, 0)

    line_centers = []
    for line in lines:
        line_centers.append(get_line_centers(line))

    plot_lines(img, line_centers)

    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    # train_model()
    # eval_model()
    # deep_eval_model()
    # generate_line_images()

    # extract_cropped_glyphs(COCO_TRAINING_DIR,
    #                        TRAIN_RAW_DIR,
    #                        TRAIN_BINARIZED_DIR,
    #                        TRAIN_CROPPED_RAW_GLYPHS_DIR)

    binarize.cnn(EVAL_RAW_DIR, EVAL_BINARIZED_DIR)

    extract_cropped_glyphs(COCO_TRAINING_DIR,
                           EVAL_RAW_DIR,
                           EVAL_BINARIZED_DIR,
                           EVAL_CROPPED_RAW_GLYPHS_DIR)
