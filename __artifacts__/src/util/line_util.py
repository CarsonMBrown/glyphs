def get_line_centers(line):
    """Returns a list of the centers of each bbox in the line."""
    return [bbox.center for bbox in line]


def unpack_lines(lines):
    return [bbox for line in lines for bbox in line]


def clean_lines(lines):
    return [[bbox for bbox in line if bbox.is_valid()] for line in lines]


def generate_line_variants(line, depth=0, merge=True, split=True):
    return generate_line_variants_at_indexes(line, depth, merge, split, allowed_indexes=None)


def generate_line_variants_at_indexes(line, depth=0, merge=True, split=True, allowed_indexes=None):
    if depth >= len(line):
        return [None]
    bbox = line[depth]
    if depth == len(line) - 1:
        if allowed_indexes is None or depth in allowed_indexes:
            return generate_bbox_variants(bbox, split)
        else:
            return [[bbox]]
    variants = []
    if allowed_indexes is None or depth in allowed_indexes:
        bbox_variants = generate_bbox_variants(bbox, split)
        for v in generate_line_variants_at_indexes(line, depth + 1, merge, split, allowed_indexes=allowed_indexes):
            for bbox_variant in bbox_variants:
                variants.append(bbox_variant + v)
    if merge and (allowed_indexes is None or depth in allowed_indexes or depth + 1 in allowed_indexes):
        bbox_variants = generate_bboxes_variants(bbox, line[depth + 1], merge)
        for v in generate_line_variants_at_indexes(line, depth + 2, merge, split, allowed_indexes=allowed_indexes):
            for bbox_variant in bbox_variants:
                variants.append((bbox_variant + v) if v is not None else bbox_variant)
    if allowed_indexes is not None and depth not in allowed_indexes:
        for v in generate_line_variants_at_indexes(line, depth + 1, merge, split, allowed_indexes=allowed_indexes):
            variants.append([bbox] + v)
    return variants


def generate_bbox_variants(bbox, split):
    variants = [[bbox]]
    if split:
        variants.append([*bbox.split()])
    return variants


def generate_bboxes_variants(bbox1, bbox2, merge):
    variants = []
    if merge:
        variants.append([bbox1.merge(bbox2)])
    return variants
