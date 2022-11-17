from statistics import mean


def get_line_centers(line):
    """Returns a list of the centers of each bbox in the line."""
    return [bbox.center for bbox in line]


def get_mean_line_dims(line):
    """Return mean (as int) bbox width and height for a line"""
    mean_width = int(mean([bbox.width for bbox in line]))
    mean_height = int(mean([bbox.height for bbox in line]))
    return mean_width, mean_height
