from math import floor, ceil

from src.util.bbox import get_bbox_pairs
from src.util.line_util import get_line_centers


def link_bboxes(bboxes):
    """
    Takes in a list of bounding boxes and generates a list of lists of
    bounding boxes representing lines
    :param bboxes: list of bounding boxes
    :return: list of lists of bounding boxes, sorted by height of left most bbox
    """
    lines = []
    bboxes = sorted(bboxes)

    # For each bounding box, generate new line or add to existing line
    for i, bbox in enumerate(bboxes):
        line = None
        # Get line that bbox is in, if it is in one
        for potential_line in lines:
            if bbox in potential_line:
                line = potential_line
                break
        # generate new line if needed, otherwise remove line from list of lines, so it can be modified out of place
        if line is None:
            line = [bbox]
        else:
            lines.remove(line)

        # For each bbox right of this bbox, add bbox to line if it is touching and in the correct y region
        if i < len(bboxes):
            for j, bbox_other in enumerate(bboxes[i + 1:], i):
                # If the other box can be added and the line remains valid, add box and find next box
                if is_line_valid(line + [bbox_other]):
                    line.append(bbox_other)
                    break

        # add line to list of lines, always maintaining an x based sort
        lines.append(line)

    # remove erroneous bounding boxes, saving removed bboxes to new lists
    oversize_bboxes = remove_oversize_bboxes(lines)
    duplicate_bboxes = remove_duplicate_bboxes(lines)
    # join lines together (if possible) after cleaning
    lines = join_lines(lines)
    return sorted_lines(lines), (oversize_bboxes, duplicate_bboxes)


def remove_bbox_intersections(lines):
    for line in lines:
        for bbox, bbox_other in get_bbox_pairs(line):
            inter_x, _ = bbox.intersections(bbox_other)
            if inter_x > 0:
                bbox.trim(floor(inter_x / 2), side="r")
                bbox_other.trim(ceil(inter_x / 2), side="l")
    return lines


def sorted_lines(lines):
    """
    Takes a list of lines and returns a sorted version of those lines
    :param lines:
    :return:
    """
    lines = [line for line in lines if line != [] and line is not None]
    for line in lines:
        line.sort()
    lines.sort(key=lambda x: x[0].y_min)
    return lines


def remove_oversize_bboxes(lines, max_sub_bboxes=1):
    """
    Removes bboxes from lines that contain the centers of more than <max_sub_bboxes> other bounding boxes
    :param lines:
    :param max_sub_bboxes:
    :return:
    """
    # if invalid input is given, return
    if max_sub_bboxes <= 0:
        return []
    # increase count by one to allow for a bounding boxes own center to be included without removing
    max_sub_bboxes += 1
    # get all bounding boxes to
    all_centers = [center for line in lines for center in get_line_centers(line)]
    # store boxes at contain <max_sub_bboxes> bbox centers inside them
    oversize_bboxes = []
    # for each line, attempt to remove all bounding boxes that are too big, maintaining line integrity
    for line in lines:
        potential_removals = []
        # for each box in a line, try to remove it if at least <max_sub_bboxes> other boxes have centers inside it
        for bbox in line:
            num = sum([bbox.contains_point(other_center) for other_center in all_centers])
            if num > max_sub_bboxes:
                potential_removals.append(bbox)
        oversize_bboxes += remove_bboxes(line, potential_removals)
    return oversize_bboxes


def remove_duplicate_bboxes(lines, iou_threshold=.75):
    """
    Remove duplicate bboxes from each line
    :param lines:
    :param iou_threshold: consider bboxes as duplicates if they have an iou > this value
    :return:
    """
    # store boxes that are removed
    duplicate_bboxes = []
    # for each line, remove duplicates from that line
    for line in lines:
        # keep bboxes that could be removed as a list of potential removals
        potential_removals = []
        for i, bbox in enumerate(line):
            potential_removals += [bbox_other for bbox_other in line[i + 1:] if
                                   bbox.iou(bbox_other) > iou_threshold]
        duplicate_bboxes += remove_bboxes(line, potential_removals)
    return duplicate_bboxes


def remove_bboxes(line, potential_removals):
    removed = []
    # only remove boxes if they can be removed without damaging the integrity of the line
    for p_r in potential_removals:
        if p_r in line and is_line_valid([bbox for bbox in line if bbox != p_r]):
            removed.append(p_r)
            line.remove(p_r)
    return removed


def join_lines(lines):
    """
    Join lines together if they intersect
    :param lines:
    :return: a new list of lines
    """
    # TODO
    # # for each line, if it has an element in common with another, join them
    # for i, line in enumerate(lines):
    #     for line_other in lines[i + 1:]:
    #         if lines_intersect(line, line_other):
    #             line += line_other
    #             lines.remove(line_other)
    # for each line, if it intersects with another line (after it in the list), join them together
    for i, line in enumerate(lines):
        for line_other in lines[i + 1:]:
            if is_line_valid(line + line_other):
                line += line_other
                lines.remove(line_other)
    return lines


def lines_intersect(line1, line2):
    """
    Join lines together if the two lines have at least one bbox in common
    :param line1:
    :param line2:
    :return:
    """
    return len([value for value in line1 if value in line2]) > 0


def is_line_valid(line, x_rel_distance_threshold=.2):
    """
    Checks that a line is valid, based on checking if the line, assuming an x-value based ordering of bboxes
    in the line, that each bbox touches (or gets quite close to) the bbox next to it
    :param line:
    :param x_rel_distance_threshold: how close bboxes need to be
    :return:
    """
    line = sorted(line)
    # for each pair of bboxes, (assuming x-value based ordering)
    for bbox, bbox_other in get_bbox_pairs(line):
        # get line centers and check if each box's center is in the others valid neighbor range
        _, in_other_y = bbox_other.contains_center(bbox, dimension_wise=True)
        _, other_in_y = bbox.contains_center(bbox_other, dimension_wise=True)
        # Both bboxes centers must be in the y range of the others
        if not in_other_y or not other_in_y:
            return False
        # check if the bboxes overlap or are close in the x direction
        iou_x, iou_y = bbox.dimensional_iou(bbox_other)
        x_rel_dist, _ = bbox.relative_edge_distance(bbox_other)
        if iou_x == 0 and x_rel_dist > x_rel_distance_threshold:
            return False  # if not, exit and return false
    return True  # valid only if no pair is invalid
