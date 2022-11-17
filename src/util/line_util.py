def get_line_centers(line):
    """Returns a list of the centers of each bbox in the line."""
    return [bbox.center for bbox in line]
