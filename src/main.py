import os.path
from functools import partial

import cv2
from numpy import arange
from tqdm import tqdm

from src.binarization import binarize
from src.bounding.bound import get_minimal_bounding_boxes_v2
from src.bounding.yolo import yolo
from src.classification.learning import nn_factory
from src.classification.learning.mnist_nn import MNISTCNN, MNISTCNN_LSTM
from src.classification.learning.resnext_lstm import ResNextLongLSTM, ResNext101LSTM
from src.classification.learning.torch_dataloader import ImageLoader
from src.classification.markov import markov
from src.evaluation.bbox_eval import remove_bbox_outliers, get_bbox_outliers, get_truth_pred_iou_tuples, \
    get_iou_metrics, get_class_metrics
from src.line_recognition.bbox_connection import link_bboxes, remove_bbox_intersections
from src.output import write_csv
from src.util.data_util import load_truth, CocoReader, write_meta
from src.util.dir_util import get_input_img_paths, init_output_dir, write_generated_bboxes, get_file_name
from src.util.glyph_util import get_classes_as_glyphs
from src.util.img_util import plot_bboxes, plot_lines, load_image
from src.util.line_util import get_line_centers, unpack_lines

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
GLYPH_DIR = os.path.join(DATASET_DIR, "glyphs")

ALL_IMAGE_DIR = os.path.join(IMAGE_DIR, "all")
ALL_RAW_DIR = os.path.join(ALL_IMAGE_DIR, "raw")
ALL_BINARIZED_DIR = os.path.join(ALL_IMAGE_DIR, "binarized")

TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train")
EVAL_IMAGE_DIR = os.path.join(IMAGE_DIR, "eval")
TEST_IMAGE_DIR = os.path.join(IMAGE_DIR, "test")

TRAIN_IMAGE_MONO_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "mono", "raw")
EVAL_IMAGE_MONO_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "mono", "raw")

TRAIN_IMAGE_MONO_BINARY_DIR = os.path.join(TRAIN_IMAGE_DIR, "mono", "binarized")
EVAL_IMAGE_MONO_BINARY_DIR = os.path.join(EVAL_IMAGE_DIR, "mono", "binarized")

TRAIN_LABEL_DIR = os.path.join(LABEL_DIR, "train")
EVAL_LABEL_DIR = os.path.join(LABEL_DIR, "eval")
TEST_LABEL_DIR = os.path.join(LABEL_DIR, "test")

TRAIN_LABEL_RAW_DIR = os.path.join(TRAIN_LABEL_DIR, "raw")
EVAL_LABEL_RAW_DIR = os.path.join(EVAL_LABEL_DIR, "raw")

TRAIN_LABEL_MONO_RAW_DIR = os.path.join(TRAIN_LABEL_DIR, "mono", "raw")
EVAL_LABEL_MONO_RAW_DIR = os.path.join(EVAL_LABEL_DIR, "mono", "raw")

TRAIN_LABEL_MONO_BINARY_DIR = os.path.join(TRAIN_LABEL_DIR, "mono", "binarized")
EVAL_LABEL_MONO_BINARY_DIR = os.path.join(EVAL_LABEL_DIR, "mono", "binarized")

TRAIN_RAW_DIR = os.path.join(TRAIN_IMAGE_DIR, "raw")
EVAL_RAW_DIR = os.path.join(EVAL_IMAGE_DIR, "raw")
TEST_RAW_DIR = os.path.join(TEST_IMAGE_DIR, "raw")

TRAIN_BINARIZED_DIR = os.path.join(TRAIN_IMAGE_DIR, "binarized")
EVAL_BINARIZED_DIR = os.path.join(EVAL_IMAGE_DIR, "binarized")
TEST_BINARIZED_DIR = os.path.join(TEST_IMAGE_DIR, "binarized")

EVAL_OUTPUT_DIR = os.path.join(EVAL_IMAGE_DIR, "output")

COCO_TRAINING_DIR = os.path.join("HomerCompTraining")
COCO_TESTING_DIR = os.path.join("HomerCompTesting")

RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "raw")
TRAIN_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "raw")
TRAIN_GENERATED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "generated")
TRAIN_GENERATED_RAW_GLYPHS_DIR = os.path.join(TRAIN_GENERATED_GLYPHS_DIR, "raw")
TRAIN_GENERATED_INTERSECTED_GLYPHS_DIR = os.path.join(TRAIN_GENERATED_GLYPHS_DIR, "intersected")
TRAIN_GENERATED_CROPPED_GLYPHS_DIR = os.path.join(TRAIN_GENERATED_GLYPHS_DIR, "cropped")
TRAIN_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "cropped_raw")
TRAIN_RAW_QUALITY_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "raw_high_quality")
TRAIN_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "binarized")
EVAL_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "raw")
EVAL_GENERATED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "generated")
EVAL_GENERATED_RAW_GLYPHS_DIR = os.path.join(EVAL_GENERATED_GLYPHS_DIR, "raw")
EVAL_GENERATED_INTERSECTED_GLYPHS_DIR = os.path.join(EVAL_GENERATED_GLYPHS_DIR, "intersected")
EVAL_GENERATED_CROPPED_GLYPHS_DIR = os.path.join(EVAL_GENERATED_GLYPHS_DIR, "cropped")
EVAL_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "cropped_raw")
BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "binarized")
EVAL_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "binarized")

RAW_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "raw")
BINARIZED_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "binarized")
ARTIFICIAL_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "artificial")

lang_file = os.path.join(DATASET_DIR, "perseus_25000.txt"), os.path.join(DATASET_DIR, "perseus_5000.txt")
quick_lang_file = os.path.join(DATASET_DIR, "perseus_5000.txt"), os.path.join(DATASET_DIR, "perseus_2000.txt")
meta_data_file = os.path.join(TRAIN_GENERATED_CROPPED_GLYPHS_DIR, "meta.csv"), \
                 os.path.join(EVAL_GENERATED_CROPPED_GLYPHS_DIR, "meta.csv")


# TRAIN_RESIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "raw")
# TRAIN_RESIZED_BINARIZED_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "resized", "binarized")
# TRAIN_OCULAR_TRAIN_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "ocular_training")
# TRAIN_OUTPUT_DIR = os.path.join(IMAGE_DIR, TRAIN_DIR, "output")
# RAW_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "raw")
# BINARIZED_OCULAR_GLYPHS_DIR = os.path.join(GLYPH_DIR, "ocular", "binarized")


def train_model():
    # nn_factory.train_model(quick_lang_file, meta_data_file,
    #                        TRAIN_GENERATED_CROPPED_GLYPHS_DIR, EVAL_GENERATED_CROPPED_GLYPHS_DIR,
    #                        ResNextLongLSTM,
    #                        epochs=200, batch_size=24, num_workers=0, resume=True,
    #                        start_epoch=1, loader=ImageLoader,
    #                        transforms=[ResNext101LSTM.transform_train_padded,
    #                                    ResNext101LSTM.transform_classify_padded],
    #                        name="generated_cropped_padded")

    nn_factory.train_model(lang_file, meta_data_file,
                           TRAIN_GENERATED_CROPPED_GLYPHS_DIR, EVAL_GENERATED_CROPPED_GLYPHS_DIR,
                           MNISTCNN_LSTM,
                           epochs=400, batch_size=64, num_workers=0, resume=True,
                           start_epoch=243, loader=ImageLoader,
                           transforms=[MNISTCNN.transform_train_2,
                                       MNISTCNN.transform_classify_2]
                           )


def eval_model():
    eval_dataset, eval_dataloader = nn_factory.generate_dataloader(os.path.join(DATASET_DIR, "perseus_25000.txt"),
                                                                   meta_data_file,
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
                                                                   meta_data_file,
                                                                   EVAL_RAW_GLYPHS_DIR, batch_size=8,
                                                                   loader=ImageLoader,
                                                                   transform=ResNext101LSTM.transform_classify)

    log_markov_chain = markov.init_markov_chain(os.path.join(DATASET_DIR, "perseus.txt"),
                                                get_classes_as_glyphs(),
                                                cache_path=os.path.join("dataset", "perseus_log.markov"),
                                                log=True,
                                                overwrite=False)
    # markov_chain = markov.init_markov_chain(os.path.join(DATASET_DIR, "perseus.txt"),
    #                                         get_classes_as_glyphs(),
    #                                         cache_path=os.path.join("dataset", "perseus.markov"),
    #                                         overwrite=False)

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


def display_lines(img, lines, *, save_path=None, wait=True):
    """
    Draws the img with lines and bboxes, saving image if given a path
    :param img: img in BGR format
    :param lines: list of lists of BBoxes
    :param save_path: optional save path
    :param wait: wait time to show on screen, 0 to wait for keypress, negative to not display
    :return: drawn image
    """
    overlay = img.copy()
    valid_bboxes = unpack_lines(lines)
    plot_bboxes(img, valid_bboxes, color=(0, 0, 0), wait=None)

    alpha = 1
    img = cv2.addWeighted(overlay, 1 - alpha, img, alpha, 0)

    line_centers = []
    for line in lines:
        line_centers.append(get_line_centers(line))

    plot_lines(img, line_centers, wait=(0 if wait else None))

    cv2.imwrite(save_path, img)
    return img


def generate_lines(img, binary_img=None, remove_intersections=False):
    valid_bboxes = generate_bboxes(img)
    lines = bboxes_to_lines(binary_img, remove_intersections, valid_bboxes)
    return lines


def generate_bboxes(img):
    bboxes = yolo.sliding_glyph_window(img)
    valid_bboxes, outlier_boxes = remove_bbox_outliers(bboxes), get_bbox_outliers(bboxes)
    return valid_bboxes


def bboxes_to_lines(binary_img, remove_intersections, bboxes):
    lines, _ = link_bboxes(bboxes)
    if remove_intersections:
        lines = remove_bbox_intersections(lines)
    if binary_img is not None:
        lines = get_minimal_bounding_boxes_v2(binary_img,
                                              lines,
                                              split=False,
                                              bboxes_in_lines=True,
                                              maintain_center=True)
    return lines


def get_input_paths(img_in_dir, binary_img_in_dir):
    color_img_list = get_input_img_paths(img_in_dir, verbose=False)
    binary_img_list = get_input_img_paths(binary_img_in_dir, verbose=False)
    # Regenerate and reload binary images as needed
    if len(binary_img_list) != len(color_img_list):
        binarize.cnn(img_in_dir, binary_img_in_dir)
        binary_img_list = get_input_img_paths(binary_img_in_dir)
    return list(zip(color_img_list, binary_img_list))


def classify_images(img_in_dir, binary_img_in_dir, img_out_dir):
    init_output_dir(img_out_dir)
    model, _ = nn_factory.load_model(ResNextLongLSTM, load_epoch=24, resume=False)
    output = []
    for color_img_path, binary_img_path in get_input_paths(img_in_dir, binary_img_in_dir):
        color_img, img_output = load_image(img_in_dir, img_out_dir, color_img_path, formattable_output=1,
                                           skip_existing=False)
        binary_img, _ = load_image(binary_img_in_dir, img_out_dir, binary_img_path, skip_existing=False,
                                   gray_scale=True)
        lines = generate_lines(color_img, binary_img=None, remove_intersections=False)
        display_lines(color_img, lines, save_path=img_output.format("_lines"), wait=False)
        nn_factory.classify(model, lines, color_img, ResNext101LSTM.transform_classify)
        output.append((color_img_path, lines))
    write_csv("out.csv", output)


def classify_lines(lines, model, img):
    nn_factory.classify(model, lines, img, ResNext101LSTM.transform_classify)


def generate_training_images(coco_dir, img_in_dir, binary_img_in_dir, out_dir, crop=False, remove_intersections=False):
    init_output_dir(out_dir)
    coco = CocoReader(coco_dir)
    meta_data = []
    remove_intersections = crop or remove_intersections
    for color_img_path, binary_img_path in get_input_paths(img_in_dir, binary_img_in_dir):
        truth_bboxes = load_truth(coco, get_file_name(color_img_path))
        color_img, img_output = load_image(img_in_dir, out_dir, color_img_path, formattable_output=1,
                                           skip_existing=False, verbose=False)
        if color_img is None:
            print("NO IMAGE: " + color_img_path)
            continue
        binary_img, _ = load_image(binary_img_in_dir, out_dir, binary_img_path, skip_existing=False,
                                   gray_scale=True, verbose=False)
        if binary_img is None:
            print("NO IMAGE: " + binary_img_path)
            continue
        pred_bboxes = unpack_lines(generate_lines(color_img,
                                                  binary_img=binary_img if crop else None,
                                                  remove_intersections=remove_intersections))
        write_generated_bboxes(
            get_truth_pred_iou_tuples(truth_bboxes, pred_bboxes),
            get_file_name(color_img_path),
            color_img,
            out_dir,
            meta_data
        )
    write_meta(meta_data, out_dir)


def generate_eval_data(coco_dir, img_in_dir, binary_img_in_dir):
    with open(os.path.join("output_data", "pipeline_metrics.csv"), mode="w") as csv_file:
        csv_file.write("intersect, crop, bbox_precision, bbox_recall, bbox_fscore, avg_iou, class_precision, "
                       "class_recall, class_fscore\n")
        model, _ = nn_factory.load_model(ResNextLongLSTM, load_epoch=24, resume=False)
        img_bboxes_pairs, input_paths, truth_bboxes = generate_img_bbox_pairs(coco_dir, img_in_dir, binary_img_in_dir)
        truth_pred_iou_tuples = []
        for crop in [True, False]:
            for intersect in [True, False]:
                with tqdm(desc="Generating Lines", total=len(input_paths)) as pbar:
                    for color_img, binary_img, template_bboxes in img_bboxes_pairs:
                        bboxes = [bbox.copy() for bbox in template_bboxes]
                        lines = bboxes_to_lines(binary_img=binary_img if crop else None,
                                                remove_intersections=intersect,
                                                bboxes=bboxes)
                        classify_lines(lines, model, color_img)
                        pred_bboxes = unpack_lines(lines)
                        truth_pred_iou_tuples += get_truth_pred_iou_tuples(truth_bboxes, pred_bboxes)
                        pbar.update(1)

                    bbox_precision, bbox_recall, bbox_fscore, avg_IOU = get_iou_metrics(truth_pred_iou_tuples)
                    class_precision, class_recall, class_fscore = get_class_metrics(truth_pred_iou_tuples)

                    csv_file.write(f"{intersect}, {crop}, {bbox_precision}, {bbox_recall}, {bbox_fscore}, {avg_IOU}, "
                                   f"{class_precision}, {class_recall}, {class_fscore}\n")


def generate_img_bbox_pairs(coco_dir, img_in_dir, binary_img_in_dir):
    coco = CocoReader(coco_dir)
    input_paths = get_input_paths(img_in_dir, binary_img_in_dir)[0:]
    img_bboxes_pairs = []
    with tqdm(desc="Bounding Boxes", total=len(input_paths)) as pbar:
        for color_img_path, binary_img_path in input_paths:
            truth_bboxes = load_truth(coco, get_file_name(color_img_path))
            color_img, _ = load_image(img_in_dir, "", color_img_path, formattable_output=1,
                                      skip_existing=False, verbose=False)
            if color_img is None:
                print("NO IMAGE: " + color_img_path)
                continue
            binary_img, _ = load_image(binary_img_in_dir, "", binary_img_path, skip_existing=False,
                                       gray_scale=True, verbose=False)
            if binary_img is None:
                print("NO IMAGE: " + binary_img_path)
                continue
            img_bboxes_pairs.append((
                color_img,
                binary_img,
                generate_bboxes(color_img)
            ))

            pbar.update(1)

    return img_bboxes_pairs, input_paths, truth_bboxes


if __name__ == '__main__':
    print("Starting...")

    train_model()
    # eval_model()
    # deep_eval_model()
    # generate_line_images()

    # generate_eval_data(COCO_TRAINING_DIR, EVAL_RAW_DIR, EVAL_BINARIZED_DIR)

    # classify(EVAL_RAW_DIR, EVAL_BINARIZED_DIR, EVAL_OUTPUT_DIR)

    # in_dir = r"C:\Users\Carson Brown\git\glyphs\dataset\display\binarization"
    # out_dir = r"C:\Users\Carson Brown\git\glyphs\output_data\eval_binarization\clustering"
    #
    # binarize.cluster(in_dir, out_dir)
