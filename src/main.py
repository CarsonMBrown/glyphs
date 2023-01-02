import os.path
from functools import partial

import cv2
import numpy
import torch
from numpy import arange, argmax, int64
from tqdm import tqdm

from src.binarization import binarize
from src.bounding.bound import get_minimal_bounding_boxes_v2
from src.bounding.yolo import yolo
from src.classification.learning import nn_factory
from src.classification.learning.mnist_nn import MNISTCNN, MNISTCNN_DEEP_LSTM
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
from src.util.line_util import get_line_centers, unpack_lines, clean_lines, generate_line_variants, \
    generate_line_variants_at_indexes

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
TRAIN_GENERATED_CROPPED_GLYPHS_DIR_V2 = TRAIN_GENERATED_CROPPED_GLYPHS_DIR + "_2"
TRAIN_GENERATED_CROPPED_GLYPHS_DIR_V3 = TRAIN_GENERATED_CROPPED_GLYPHS_DIR + "_3"
TRAIN_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "cropped_raw")
TRAIN_RAW_QUALITY_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "raw_high_quality")
TRAIN_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "train", "binarized")
EVAL_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "raw")
EVAL_GENERATED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "generated")
EVAL_GENERATED_RAW_GLYPHS_DIR = os.path.join(EVAL_GENERATED_GLYPHS_DIR, "raw")
EVAL_GENERATED_INTERSECTED_GLYPHS_DIR = os.path.join(EVAL_GENERATED_GLYPHS_DIR, "intersected")
EVAL_GENERATED_CROPPED_GLYPHS_DIR = os.path.join(EVAL_GENERATED_GLYPHS_DIR, "cropped")
EVAL_GENERATED_CROPPED_GLYPHS_DIR_V2 = EVAL_GENERATED_CROPPED_GLYPHS_DIR + "_2"
EVAL_GENERATED_CROPPED_GLYPHS_DIR_V3 = EVAL_GENERATED_CROPPED_GLYPHS_DIR + "_3"
EVAL_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "cropped_raw")
TEST_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "test", "raw")
TEST_GENERATED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "test", "generated")
TEST_GENERATED_RAW_GLYPHS_DIR = os.path.join(TEST_GENERATED_GLYPHS_DIR, "raw")
TEST_GENERATED_INTERSECTED_GLYPHS_DIR = os.path.join(TEST_GENERATED_GLYPHS_DIR, "intersected")
TEST_GENERATED_CROPPED_GLYPHS_DIR = os.path.join(TEST_GENERATED_GLYPHS_DIR, "cropped")
TEST_GENERATED_CROPPED_GLYPHS_DIR_V2 = TEST_GENERATED_CROPPED_GLYPHS_DIR + "_2"
TEST_GENERATED_CROPPED_GLYPHS_DIR_V3 = TEST_GENERATED_CROPPED_GLYPHS_DIR + "_3"
TEST_CROPPED_RAW_GLYPHS_DIR = os.path.join(GLYPH_DIR, "test", "cropped_raw")
BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "binarized")
EVAL_BINARIZED_GLYPHS_DIR = os.path.join(GLYPH_DIR, "eval", "binarized")

RAW_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "raw")
BINARIZED_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "binarized")
ARTIFICIAL_TEMPLATE_GLYPHS_DIR = os.path.join(GLYPH_DIR, "templates", "artificial")

lang_file = os.path.join(DATASET_DIR, "perseus_25000.txt"), os.path.join(DATASET_DIR, "perseus_5000.txt")
quick_lang_file = os.path.join(DATASET_DIR, "perseus_5000.txt"), os.path.join(DATASET_DIR, "perseus_2000.txt")
meta_data_file = os.path.join(TRAIN_GENERATED_CROPPED_GLYPHS_DIR, "meta.csv"), \
                 os.path.join(EVAL_GENERATED_CROPPED_GLYPHS_DIR, "meta.csv")


def train_model(train_dir, eval_dir, resume=False, start_epoch=0, epochs=100, batch_size=24):
    """
    Train the best network with the given parameters
    """
    nn_factory.train_model(
        lang_file, meta_data_file, train_dir, eval_dir, ResNextLongLSTM, epochs=epochs, batch_size=batch_size,
        num_workers=0, resume=resume, start_epoch=start_epoch, loader=ImageLoader,
        transforms=[ResNext101LSTM.transform_train,
                    ResNext101LSTM.transform_classify],
        loss_fn=torch.nn.CrossEntropyLoss
    )


def eval_model(eval_dir, batch_size=24, start_epoch=0, epochs=100):
    """Generates eval data for each epoch of the best network"""
    # generate data loader
    eval_dataset, eval_dataloader = nn_factory.generate_dataloader(
        lang_file[-1],
        meta_data_file[-1],
        eval_dir, batch_size=batch_size,
        loader=ImageLoader,
        transform=ResNext101LSTM.transform_classify
    )

    print("Epoch, Precision, Recall, FScore")
    # for each epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        # load model at epoch
        model, _ = nn_factory.load_model(
            ResNext101LSTM, load_epoch=epoch, dataset=eval_dataset, resume=False)
        # get eval metrics for model
        avg_precision, avg_recall, avg_fscore = nn_factory.eval_model(model, eval_dataloader, average="weighted",
                                                                      seed=0)
        print(f"{epoch}, {avg_precision}, {avg_recall}, {avg_fscore}")
        # delete model from memory
        del model


def eval_model_with_language_model(eval_dir, epoch, batch_size=24):
    # generate data loader
    eval_dataset, eval_dataloader = nn_factory.generate_dataloader(
        lang_file[-1],
        meta_data_file[-1],
        eval_dir, batch_size=batch_size,
        loader=ImageLoader,
        transform=ResNext101LSTM.transform_classify
    )

    log_markov_chain = markov.init_markov_chain(
        os.path.join(DATASET_DIR, "perseus.txt"),
        get_classes_as_glyphs(),
        cache_path=os.path.join("dataset", "perseus_log.markov"),
        log=True, overwrite=False
    )

    # load model at epoch
    model, _ = nn_factory.load_model(ResNextLongLSTM, load_epoch=epoch, dataset=eval_dataset, resume=False)

    print("uncertainty, precision, recall, f1score")

    for uncertainty in arange(0, 1.01, 0.01):
        # round uncertainty to deal with floating point errors
        uncertainty = round(uncertainty, 2)
        # get confusion matrix, precision, recall, and fscore
        confusion_matrix, (precision, recall, fscore, _) = nn_factory.model_confusion_matrix(
            model, eval_dataloader,
            display_cm=False, seed=0,
            top_k=1,
            prediction_modifier=partial(
                markov.top_n_markov_optimization,
                log_markov_chain,
                n=2,
                uncertainty_threshold=uncertainty
            ))
        print(f"{uncertainty}, {precision}, {recall}, {fscore}")


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


def generate_lines(img, *, binary_img=None, remove_intersections=False, inner_sliding_window=True,
                   bbox_gif_export_path=None):
    valid_bboxes = generate_bboxes(img, inner_sliding_window=inner_sliding_window,
                                   bbox_gif_export_path=bbox_gif_export_path)
    lines = bboxes_to_lines(binary_img, remove_intersections, valid_bboxes)
    return lines


def generate_line_image(img, *, binary_img=None, remove_intersections=False, inner_sliding_window=True, save_path=None,
                        wait=True, bbox_gif_export_path=None):
    lines = generate_lines(img, binary_img=binary_img,
                           remove_intersections=remove_intersections,
                           inner_sliding_window=inner_sliding_window,
                           bbox_gif_export_path=bbox_gif_export_path)
    display_lines(img, lines, save_path=save_path, wait=wait)


def generate_line_images(img_in_dir, binary_img_in_dir, save_path, remove_intersections=False,
                         inner_sliding_window=True):
    init_output_dir(save_path)
    for color_img_path, binary_img_path in get_input_paths(img_in_dir, binary_img_in_dir):
        x = get_image_output_tuple(img_in_dir, binary_img_in_dir, color_img_path, binary_img_path, save_path,
                                   verbose=False, formattable_output=0)
        print(color_img_path)
        if x is None:
            continue
        color_img, binary_img, img_output = x
        generate_line_image(color_img, binary_img=binary_img, save_path=img_output,
                            remove_intersections=remove_intersections, inner_sliding_window=inner_sliding_window,
                            wait=False, bbox_gif_export_path=None)


def get_image_output_tuple(img_in_dir, binary_img_in_dir, color_img_path, binary_img_path, save_path, verbose=False,
                           formattable_output=0):
    color_img, img_output = load_image(img_in_dir, save_path, color_img_path, formattable_output=formattable_output,
                                       skip_existing=False, verbose=verbose)
    if color_img is None:
        print("NO COLOR IMAGE: " + color_img_path)
        return None
    if binary_img_in_dir is not None:
        binary_img, _ = load_image(binary_img_in_dir, save_path, binary_img_path, skip_existing=False,
                                   gray_scale=True, verbose=verbose)
        if binary_img is None:
            print("NO BINARY IMAGE: " + binary_img_path)
            return None
        return color_img, binary_img, img_output
    return color_img, None, img_output


def get_image_output_tuples(img_in_dir, save_path, *, binary_img_in_dir=None, verbose=False,
                            formattable_output=0):
    images_output_pairs = []
    for color_img_path, binary_img_path in get_input_paths(img_in_dir, binary_img_in_dir):
        x = get_image_output_tuple(img_in_dir, binary_img_in_dir, color_img_path, binary_img_path, save_path,
                                   verbose, formattable_output)
        if x is not None:
            images_output_pairs.append((x[0], color_img_path, x[1], x[2]))
    return images_output_pairs


def generate_bboxes(img, *, inner_sliding_window=True, bbox_gif_export_path=None, confidence_min=None, cache_name=None,
                    duplicate_threshold=None):
    """Generate the bounding boxes in the passed in image"""
    bboxes = yolo.sliding_glyph_window(img, inner_sliding_window=inner_sliding_window,
                                       bbox_gif_export_path=bbox_gif_export_path,
                                       confidence_min=confidence_min, cache_name=cache_name,
                                       duplicate_threshold=duplicate_threshold)
    valid_bboxes, outlier_boxes = remove_bbox_outliers(bboxes), get_bbox_outliers(bboxes)
    return valid_bboxes


def bboxes_to_lines(binary_img, remove_intersections, bboxes):
    """Bind bboxes into lines"""
    lines, _ = link_bboxes(bboxes)
    if remove_intersections:
        lines = remove_bbox_intersections(lines)
    if binary_img is not None:
        lines = get_minimal_bounding_boxes_v2(binary_img,
                                              lines,
                                              split=False,
                                              bboxes_in_lines=True,
                                              maintain_center=True)
    return clean_lines(lines)


def get_input_paths(img_in_dir, binary_img_in_dir):
    color_img_list = get_input_img_paths(img_in_dir, verbose=False)
    if binary_img_in_dir is not None:
        binary_img_list = get_input_img_paths(binary_img_in_dir, verbose=False)
    else:
        binary_img_list = None
    # Regenerate and reload binary images as needed
    if binary_img_list is not None and len(binary_img_list) != len(color_img_list):
        binarize.cnn(img_in_dir, binary_img_in_dir)
        binary_img_list = get_input_img_paths(binary_img_in_dir)
    if binary_img_list is None:
        binary_img_list = [None] * len(color_img_list)
    return list(zip(color_img_list, binary_img_list))


def classify_lines(lines, model, img, transform, flip_channels=True):
    """Classify lines, flipping RGB to BGR or back if needed"""
    if flip_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nn_factory.classify(model, lines, img, transform)
    return lines


def classify_lines_with_variants(lines, model, img, transform, softmax=False, flip_channels=True):
    """Classify lines and their variants, maximizing classification confidence score."""
    if flip_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_lines = []
    for line in lines:
        potential_lines = generate_line_variants(line)
        best = argmax([nn_factory.classify(model, potential_line, img, transform, softmax) for
                       potential_line in potential_lines])
        if not isinstance(best, int64):
            print(type(best))
            best = best[0]
        new_lines.append(potential_lines[best])
    return new_lines


def classify_lines_with_adaptive_variants(lines, model, img, transform, softmax=False, variants_per_line=1,
                                          flip_channels=True):
    """Classify lines and their variants, maximizing classification confidence score."""
    if flip_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_lines = []
    for line in lines:
        log_probs = nn_factory.classify(model, line, img, transform, softmax)
        indexes_to_vary = list(numpy.argsort(numpy.array(log_probs)))[-variants_per_line:]
        potential_lines = generate_line_variants_at_indexes(line, allowed_indexes=indexes_to_vary)

        max_prob = None
        best_line = None
        for potential_line in potential_lines:
            log_probs = nn_factory.classify(model, potential_line, img, transform, softmax)
            log_prob = sum(log_probs)
            if max_prob is None or max_prob < log_prob:
                max_prob = log_prob
                best_line = potential_line

        new_lines.append(best_line)
    return new_lines


def generate_training_images(coco_dir, img_in_dir, binary_img_in_dir, out_dir, *, crop=False,
                             remove_intersections=False):
    """
    Generate cropped or otherwise images from YOLO to train/eval on
    """
    init_output_dir(out_dir)
    coco = CocoReader(coco_dir)
    meta_data = []
    remove_intersections = crop or remove_intersections
    for color_img_path, binary_img_path in get_input_paths(img_in_dir, binary_img_in_dir):
        truth_bboxes = load_truth(coco, get_file_name(color_img_path))
        x = get_image_output_tuple(img_in_dir,
                                   binary_img_in_dir,
                                   color_img_path,
                                   binary_img_path,
                                   out_dir,
                                   formattable_output=1)
        if x is None:
            continue
        color_img, binary_img, img_output = x
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
    """
    Generate full pipelin evaluation data / metrics
    """
    with open(os.path.join("output_data", "pipeline_metrics.csv"), mode="w") as csv_file:
        csv_file.write(
            "confidence_min, duplicate_threshold, inner_sliding_window, intersect, crop, complex_nn, "
            "vary_lines, make_lines, bbox_precision, bbox_recall, "
            "bbox_fscore, avg_iou, class_precision, class_recall, class_fscore\n"
        )
        cropped_model, _ = nn_factory.load_model(MNISTCNN_DEEP_LSTM, load_epoch=773, resume=False)
        full_size_model, _ = nn_factory.load_model(ResNextLongLSTM, load_epoch=24, resume=False)

        for inner_window in [False]:
            if inner_window:
                min_confidence_val, max_confidence_val = .6, .8
            else:
                min_confidence_val, max_confidence_val = .6, .8
            for confidence_min in arange(min_confidence_val, max_confidence_val, .025):
                confidence_min = round(confidence_min, 3)
                if inner_window:
                    min_duplicate_val, max_duplicate_val = .1, .3
                else:
                    min_duplicate_val, max_duplicate_val = .1, .3
                for duplicate_threshold in arange(min_duplicate_val, max_duplicate_val, .025):
                    duplicate_threshold = round(duplicate_threshold, 3)
                    img_bboxes_dicts = generate_img_bbox_dicts(coco_dir, img_in_dir,
                                                               binary_img_in_dir,
                                                               inner_sliding_window=inner_window,
                                                               confidence_min=confidence_min,
                                                               duplicate_threshold=duplicate_threshold)
                    crop, intersect, complex_nn, vary_lines, make_lines = False, False, True, False, True
                    for crop, intersect in n_booleans(2):
                        truth_pred_iou_tuples = []
                        with tqdm(desc=f"Generating Lines w/ confidence:{confidence_min}",
                                  total=len(img_bboxes_dicts)) as pbar:
                            for i, img_bboxes_dict in enumerate(img_bboxes_dicts):

                                bboxes = [bbox.copy() for bbox in img_bboxes_dict["pred"]]
                                truth_bboxes = [bbox.copy() for bbox in img_bboxes_dict["truth"]]
                                if make_lines:
                                    lines = bboxes_to_lines(binary_img=img_bboxes_dict["binary"] if crop else None,
                                                            remove_intersections=intersect,
                                                            bboxes=bboxes)
                                else:
                                    lines = [[bbox] for bbox in bboxes]
                                if complex_nn:
                                    model, transform = full_size_model, ResNext101LSTM.transform_classify_padded
                                else:
                                    model, transform = cropped_model, MNISTCNN.transform_classify_2

                                if vary_lines:
                                    lines = classify_lines_with_adaptive_variants(lines, model,
                                                                                  img_bboxes_dict["color"], transform,
                                                                                  softmax=complex_nn,
                                                                                  flip_channels=False)
                                else:
                                    lines = classify_lines(lines, model, img_bboxes_dict["color"], transform,
                                                           flip_channels=False)

                                pred_bboxes = unpack_lines(lines)
                                truth_pred_iou_tuples += get_truth_pred_iou_tuples(truth_bboxes, pred_bboxes)
                                pbar.update(1)

                            bbox_precision, bbox_recall, bbox_fscore, avg_IOU = \
                                get_iou_metrics(truth_pred_iou_tuples)
                            class_precision, class_recall, class_fscore = get_class_metrics(
                                truth_pred_iou_tuples)

                            csv_file.write(
                                f"{confidence_min}, {duplicate_threshold}, {inner_window}, "
                                f"{intersect}, {crop}, {complex_nn}, "
                                f"{vary_lines}, {make_lines}, {bbox_precision}, {bbox_recall}, "
                                f"{bbox_fscore}, {avg_IOU}, {class_precision}, {class_recall}, {class_fscore}\n")


def generate_img_bbox_dicts(coco_dir, img_in_dir, binary_img_in_dir, *, inner_sliding_window=True,
                            confidence_min=None, duplicate_threshold=None):
    """
    Given a coco file and an image directory path, generate a list of,
    (image, binary image, predicted bounding box, truth bounding box) tuples
    """
    coco = CocoReader(coco_dir)
    input_paths = get_input_paths(img_in_dir, binary_img_in_dir)[0:]
    img_bboxes_dicts = []
    with tqdm(desc="Bounding Boxes", total=len(input_paths)) as pbar:
        for color_img_path, binary_img_path in input_paths:
            truth_bboxes = load_truth(coco, get_file_name(color_img_path))
            color_img, _ = load_image(img_in_dir, "", color_img_path, formattable_output=1,
                                      skip_existing=False, verbose=False)
            if color_img is None:
                print("NO IMAGE: " + color_img_path)
                continue

            if binary_img_path is not None:
                binary_img, _ = load_image(binary_img_in_dir, "", binary_img_path, skip_existing=False,
                                           gray_scale=True, verbose=False)
                if binary_img is None:
                    print("NO IMAGE: " + binary_img_path)
                    continue
            else:
                binary_img = None

            img_bboxes_dicts.append({
                "color": cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB),
                "binary": binary_img,
                "pred": generate_bboxes(color_img, inner_sliding_window=inner_sliding_window,
                                        confidence_min=confidence_min,
                                        cache_name=color_img_path, duplicate_threshold=duplicate_threshold),
                "truth": truth_bboxes
            })
            pbar.update(1)

    return img_bboxes_dicts


def n_booleans(n):
    """
    Generate all the lists of n elements that can be made of booleans
    :param n: number of elements to have in the lists being generated
    :return a list of lists, where each list is made up of n booleans, s.t each ordered list of booleans is unique
    """
    return n_choices(n, [True, False])


def n_choices(n, choices):
    """
    From a given list of choices, generate all the lists of n elements that can be made by picking an element
    from the list of choices at each index in the new list
    :param n: number of elements to have in the lists being generated
    :param choices: list of choices for each element in the new lists
    :return a list of lists, where each list is made up of n elements from the choices list s.t. the ordered lists that
    are returned are unique
    """
    x = [[c] for c in choices]
    if n == 1:
        return x
    for _ in range(n - 1):
        x = [C + [c] for c in choices for C in x]
    return x


def transcribe_images(img_in_dir, img_out_dir, *, binary_img_in_dir=None):
    """Output transcription for all images"""
    init_output_dir(img_out_dir)
    img_path_tuples = get_image_output_tuples(img_in_dir, img_out_dir, formattable_output=1,
                                              binary_img_in_dir=binary_img_in_dir)
    model, _ = nn_factory.load_model(ResNextLongLSTM, load_epoch=24, resume=False)
    output = []
    for img, img_path, binary_img, save_path in tqdm(img_path_tuples, desc="Transcribing Image"):
        lines = generate_lines(img, binary_img=binary_img, remove_intersections=False)
        display_lines(img, lines, save_path=save_path.format("_lines"), wait=False)
        lines = classify_lines(lines, model, img, ResNext101LSTM.transform_classify)
        output.append((img_path, lines))
    write_csv("out.csv", output)


if __name__ == '__main__':
    print("Starting...")

    # eval_model()
    # deep_eval_model()
    # generate_line_images(EVAL_RAW_DIR, EVAL_BINARIZED_DIR,
    #                      os.path.join(EVAL_OUTPUT_DIR, "intersected_cropped_no_window"),
    #                      remove_intersections=True, inner_sliding_window=False)

    # if False:
    #     generate_line_image(
    #         cv2.imread(
    #             r"C:\Users\Carson Brown\git\glyphs\dataset\images\eval\raw\Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.jpg"),
    #         binary_img=cv2.imread(
    #             r"C:\Users\Carson Brown\git\glyphs\dataset\images\eval\binarized\Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.png",
    #             cv2.IMREAD_GRAYSCALE),
    #         remove_intersections=True, inner_sliding_window=False,
    #         wait=False,
    #         bbox_gif_export_path=r"C:\Users\Carson Brown\git\glyphs\output_data\sliding_window\Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8")
    #
    # generate_line_image(
    #     cv2.imread(
    #         r"C:\Users\Carson Brown\git\glyphs\dataset\images\eval\raw\Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.jpg"),
    #     binary_img=cv2.imread(
    #         r"C:\Users\Carson Brown\git\glyphs\dataset\images\eval\binarized\Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.png",
    #         cv2.IMREAD_GRAYSCALE),
    #     save_path=r"C:\Users\Carson Brown\git\glyphs\output_data\line_generation\Bodleian_Library_MS_Gr_class_a_1_P_1_10_00008_frame_8.png",
    #     remove_intersections=True, inner_sliding_window=False,
    #     wait=False)

    # generate_training_images(COCO_TRAINING_DIR, TRAIN_RAW_DIR, TRAIN_BINARIZED_DIR,
    #                          TRAIN_GENERATED_CROPPED_GLYPHS_DIR_V3, remove_intersections=False, crop=False)
    # generate_training_images(COCO_TRAINING_DIR, EVAL_RAW_DIR, EVAL_BINARIZED_DIR, EVAL_GENERATED_CROPPED_GLYPHS_DIR_V3,
    #                          remove_intersections=False, crop=False)
    # generate_training_images(COCO_TRAINING_DIR, TEST_RAW_DIR, TEST_BINARIZED_DIR, TEST_GENERATED_CROPPED_GLYPHS_DIR_V3,
    #                          remove_intersections=False, crop=False)
    transcribe_images(r"temp\test_input_dir", r"temp\test_output_dir")
    # generate_eval_data(COCO_TRAINING_DIR, EVAL_RAW_DIR, EVAL_BINARIZED_DIR)
    # train_model()
