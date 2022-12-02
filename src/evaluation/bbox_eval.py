from statistics import mean

from sklearn.metrics import precision_recall_fscore_support

from src.util.bbox import get_mean_dims


def get_truth_pred_iou_tuples(truth_bboxes, pred_bboxes):
    """
    Pair truths and predictions based on iou. Returns a list of containing tuples such that every truth and predication
    is included in a tuple, even if no pair is found.
    :param truth_bboxes:
    :param pred_bboxes:
    :return: [(truth, prediction, iou)] with truth or predication = None if no pair was found
    """
    # get truth bbox with highest iou for each predication and make pair
    truth_pred_pairs = get_optimal_pairs(truth_bboxes, pred_bboxes)
    # pair all remaining truths with a None value and an iou of 0, denoting no valid pred was found
    for truth in truth_bboxes:
        truth_pred_pairs.append((truth, None, 0))
    return truth_pred_pairs


def get_optimal_pairs(truth_bboxes, pred_bboxes):
    truth_pred_pairs = []
    for pred in pred_bboxes:
        if len(truth_bboxes) == 0:
            break
        truth, iou = pred.get_pair(truth_bboxes)
        truth_pair, _ = truth.get_pair(pred_bboxes)
        # only pair if iou > 0 and optimal pair for truth is pred, otherwise add pred with None truth and iou of 0
        if truth_pair == pred:
            truth_bboxes.remove(truth)
            truth_pred_pairs.append((truth, pred, iou))
        else:
            truth_pred_pairs.append((None, pred, 0))
    return truth_pred_pairs


def get_iou_metrics(truth_pred_tuples):
    """
    Get bounding box metrics
    :param truth_pred_tuples: [(truth, prediction, iou)] with truth or predication = None if no pair was found
    :return: precision, recall, fscore, avg_IOU across all predications
    """
    tp = len([1 for _, _, iou in truth_pred_tuples if iou > 0])
    fp = len([1 for truth, _, iou in truth_pred_tuples if truth is None])
    fn = len([1 for _, pred, iou in truth_pred_tuples if pred is None])
    avg_IOU = mean([iou for truth, _, iou in truth_pred_tuples if truth is not None] + [0])
    precision, recall = tp / (tp + fp) if (tp + fp) > 0 else 0, tp / (tp + fn) if (tp + fn) > 0 else 0
    fscore = ((2 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return precision, recall, fscore, avg_IOU


def get_class_metrics(truth_pred_tuples):
    """
    Get classification metrics
    :param truth_pred_tuples: [(truth, prediction, iou)] with truth or predication = None if no pair was found
    :return: precision, recall, fscore
    """
    valid_truth_pred_pairs = [(truth.get_class(), pred.get_class()) for truth, pred, iou in truth_pred_tuples if
                              iou > 0]
    truths = [truth for truth, pred in valid_truth_pred_pairs]
    predictions = [pred for truth, pred in valid_truth_pred_pairs]
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(truths, predictions, average="weighted", zero_division=0)
    return precision, recall, fscore


def mean_iou(bbox, bboxes):
    return mean([bbox.iou(other) for other in bboxes])


def get_unique_bboxes(bboxes, iou_threshold=.8):
    # only keep unique boxes
    unique_bboxes = []
    for i, a in enumerate(bboxes):
        duplicate = False
        for b in bboxes[i + 1:]:
            if a.iou(b) > iou_threshold:
                duplicate = True
        if not duplicate:
            unique_bboxes.append(a)
    return unique_bboxes


def get_non_enclosed_bboxes(bboxes):
    return [bbox for bbox in bboxes if len(bbox.get_enclosing(bboxes)) == 0]


def remove_bbox_outliers(bboxes, *, x_min_percent=.5, x_max_percent=2, y_min_percent=.5, y_max_percent=2):
    """
    Removes all bounding boxes where the width or height of the bounding box is more/less than the allowed
    max/min percentage of the mean width/height.
    :param bboxes:
    :param x_min_percent:
    :param x_max_percent:
    :param y_min_percent:
    :param y_max_percent:
    :return:
    """
    dx_mean, dy_mean = get_mean_dims(bboxes)
    return [
        bbox for bbox in bboxes if
        dx_mean * x_min_percent <= bbox.width <= dx_mean * x_max_percent or
        dy_mean * y_min_percent <= bbox.height <= dy_mean * y_max_percent
    ]


def get_bbox_outliers(bboxes, *, x_min_percent=.5, x_max_percent=2, y_min_percent=.5, y_max_percent=2):
    """
    Gets all bounding boxes where the width or height of the bounding box is more/less than the allowed
    max/min percentage of the mean width/height.
    :param bboxes:
    :param x_min_percent:
    :param x_max_percent:
    :param y_min_percent:
    :param y_max_percent:
    :return:
    """
    non_outliers = remove_bbox_outliers(bboxes, x_min_percent=x_min_percent, x_max_percent=x_max_percent,
                                        y_min_percent=y_min_percent, y_max_percent=y_max_percent)
    return [bbox for bbox in bboxes if bbox not in non_outliers]
