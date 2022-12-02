def get_line_centers(line):
    """Returns a list of the centers of each bbox in the line."""
    return [bbox.center for bbox in line]


def unpack_lines(lines):
    return [bbox for line in lines for bbox in line]


def clean_lines(lines):
    return [[bbox for bbox in line if bbox.is_valid()] for line in lines]


def generate_line_variants(line, depth=0, maintain=True, merge=True, split=True):
    if depth >= len(line):
        return [None]
    bbox = line[depth]
    variants = []
    if depth == len(line) - 1:
        if split:
            variants += [[*bbox.split()]]
        if maintain:
            variants += [[bbox]]
        return variants
    for v in generate_line_variants(line, depth + 1, maintain, merge, split):
        if maintain:
            variants.append([bbox] + v)
        if split:
            variants.append([*bbox.split()] + v)
    if merge:
        for v in generate_line_variants(line, depth + 2, maintain, merge, split):
            if v is not None:
                variants.append([bbox.merge(line[depth + 1])] + v)
            else:
                variants.append([bbox.merge(line[depth + 1])])
    return variants
