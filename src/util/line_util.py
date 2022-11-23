def get_line_centers(line):
    """Returns a list of the centers of each bbox in the line."""
    return [bbox.center for bbox in line]


def unpack_lines(lines):
    return [bbox for line in lines for bbox in line]
