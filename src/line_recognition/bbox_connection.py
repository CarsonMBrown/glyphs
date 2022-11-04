from src.util.bbox_util import point_in_bbox, dimensional_iou, bbox_center, relative_edge_distance, iou


def link_bboxes(bboxes):
    """
    Takes in a list of pascal formatted (x_min, y_min, x_max, y_max) bounding boxes and generates a list of lists of
    bounding boxes representing lines
    :param bboxes: list of bounding boxes in pascal form (x_min, y_min, x_max, y_max)
    :return: list of lists of bounding boxes, sorted by height of left most bbox
    """
    lines = []
    bboxes = sorted(bboxes, key=lambda x: x[0])

    for i, bbox in enumerate(bboxes):
        line = None
        # Get line that bbox is in, if it is in one
        for potential_line in lines:
            if bbox in potential_line:
                line = potential_line
                break
        if line is None:
            line = [bbox]
        else:
            lines.remove(line)

        c = bbox_center(bbox)
        # For each bbox right of this bbox, add bbox to line if it is touching and in the correct y region
        if i < len(bboxes):
            for j, bbox_other in enumerate(bboxes[i + 1:], i):
                # If the other box can be added and the line remains valid, add box and find next box
                if is_line_valid(line + [bbox_other]):
                    line.append(bbox_other)
                    break

        lines.append(line)

    oversize_bboxes = remove_oversize_bboxes(lines)
    duplicate_bboxes = remove_duplicate_bboxes(lines)
    join_lines(lines)
    return sorted_lines(lines), (oversize_bboxes, duplicate_bboxes)


def sorted_lines(lines):
    lines = [line for line in lines if line != [] and line is not None]
    for line in lines:
        line.sort(key=lambda bbox: bbox[0])
    lines.sort(key=lambda x: x[0][1])
    return lines


def remove_oversize_bboxes(lines):
    all_bboxes = [bbox for line in lines for bbox in line]
    oversize_bboxes = []
    for line in lines:
        potential_removals = []
        # for each box in a line, try to remove it if at least two other boxes have centers inside of it
        for bbox in line:
            num = sum([point_in_bbox(bbox_center(other), bbox) for other in all_bboxes])
            if num > 2:
                potential_removals.append(bbox)
        # only remove boxes if they can be removed without damaging the integrity of the line
        for p_r in potential_removals:
            if is_line_valid([bbox for bbox in line if bbox != p_r]):
                oversize_bboxes.append(p_r)
                line.remove(p_r)
    return oversize_bboxes


def remove_duplicate_bboxes(lines):
    duplicate_bboxes = []
    for line in lines:
        potential_removals = []
        for i, bbox in enumerate(line):
            potential_removals += [bbox_other for bbox_other in line[i + 1:] if
                                   iou(bbox, bbox_other) > .75]
        # only remove boxes if they can be removed without damaging the integrity of the line
        for p_r in potential_removals:
            if is_line_valid([bbox for bbox in line if bbox != p_r]):
                duplicate_bboxes.append(p_r)
                line.remove(p_r)
    return duplicate_bboxes


def join_lines(lines):
    for i, line in enumerate(lines):
        for line_other in lines[i + 1:]:
            if lines_intersect(line, line_other):
                line += line_other
                lines.remove(line_other)
    return lines


def lines_intersect(line1, line2):
    return len([value for value in line1 if value in line2]) > 0


def is_line_valid(line):
    # for each pair of bboxes, (assuming x based order)
    for bbox, bbox_other in zip(line[:-1], line[1:]):
        # get line centers
        c = bbox_center(bbox)
        c_other = bbox_center(bbox_other)
        # check if each box's center is in the others valid neighbor range
        _, in_other_y = point_in_bbox(c, bbox_other, dimension_wise=True)
        _, other_in_y = point_in_bbox(c_other, bbox, dimension_wise=True)
        if not in_other_y or not other_in_y:
            return False  # if not, exit and return false
        # check if the bboxes overlap or are close in the x direction
        iou_x, iou_y = dimensional_iou(bbox, bbox_other)
        x_rel_distance, y_rel_distance = relative_edge_distance(bbox, bbox_other)
        if iou_x == 0 or x_rel_distance > .05:
            return False  # if not, exit and return false
    return True  # valid only if no pair is invalid
