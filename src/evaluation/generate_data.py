import os

from src.bounding import bound
from src.bounding.connected_components import bound_and_render
from src.evaluation.bbox_eval import get_truth_pred_iou_tuples, get_iou_metrics
from src.main import generate_img_bbox_pairs, n_choices


def eval_connected_components_bounding(coco_dir, binary_img_in_dir):
    with open(os.path.join("output_data", "cc_bounding_metrics.csv"), mode="w") as csv_file:
        csv_file.write("internal_centroids, internal_regions, bbox_precision, bbox_recall, bbox_fscore, avg_iou\n")
        img_bboxes_pairs, template_truth_bboxes = generate_img_bbox_pairs(coco_dir, binary_img_in_dir,
                                                                          binary_img_in_dir)
        for centroid, region in n_choices(2, [True, False]):
            truth_pred_iou_tuples = []
            for i, (color_img, binary_img, template_bboxes) in enumerate(img_bboxes_pairs):
                pred_bboxes = bound.get_connected_component_bounding_boxes(binary_img, centroid, region)
                truth_bboxes = [bbox.copy() for bbox in template_truth_bboxes[i]]
                truth_pred_iou_tuples += get_truth_pred_iou_tuples(truth_bboxes, pred_bboxes)

            bbox_precision, bbox_recall, bbox_fscore, avg_IOU = get_iou_metrics(truth_pred_iou_tuples)
            csv_file.write(f"{centroid}, {region}, {bbox_precision}, {bbox_recall}, {bbox_fscore}, {avg_IOU}\n")


# eval_connected_components_bounding(COCO_TRAINING_DIR, TEST_BINARIZED_DIR)
bound_and_render("dataset/display/binarization/cnn_masked", "output_data/eval_bounding")
