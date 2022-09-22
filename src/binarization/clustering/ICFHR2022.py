import json
from collections import defaultdict

dir = '../../../HomerCompTraining/'


# Attributes of annotations:
# * id
# * image_id
# * category_id
# * bbox
# ? tags
# ? seg_id
def ann_id(a):
    return a['id']


def ann_img_id(a):
    return a['image_id']


def ann_cat_id(a):
    return a['category_id']


def ann_x(a):
    return a['bbox'][0]


def ann_y(a):
    return a['bbox'][1]


def ann_w(a):
    return a['bbox'][2]


def ann_h(a):
    return a['bbox'][3]


def ann_basetype(a):
    return a['tags']['BaseType'][0]


def ann_footmarktype(a):
    if 'FootMarkType' in a['tags']:
        return a['tags']['FootMarkType'][+0]
    else:
        return ''


# Attributes of categories:
# * id
# * name
def cat_id(c):
    return c['id']


def cat_name(c):
    return c['name']


# Attributes of figures:
# * id
# * file_name
def img_id(i):
    return i['id']


def img_file(i):
    return i['file_name']


class ICFHR2022:
    def __init__(self):
        with open(dir + 'HomerCompTrainingReadCoco.json', encoding='utf-8') as json_file:
            admin = json.load(json_file)
        self.annotations = admin['annotations']  # character bounding boxes
        self.categories = admin['categories']  # character types
        self.images = admin['figures']  # figures

        self.id_to_ann = {}
        for a in self.annotations:
            self.id_to_ann[ann_id(a)] = a

        self.id_to_cat = {}
        for c in self.categories:
            self.id_to_cat[cat_id(c)] = c

        self.id_to_img = {}
        for i in self.images:
            self.id_to_img[img_id(i)] = i

        self.img_to_ann = defaultdict(list)
        for a in self.annotations:
            self.img_to_ann[ann_img_id(a)].append(a)

        self.name_to_ann = defaultdict(list)
        for a in self.annotations:
            id = ann_cat_id(a)
            cat = self.id_to_cat[id]
            name = cat_name(cat)
            self.name_to_ann[name].append(a)

    def img_ids(self):
        return sorted(self.id_to_img.keys())

    def ann_name(self, a):
        return cat_name(self.id_to_cat[ann_cat_id(a)])

    def ann_class(self, a):
        return self.ann_name(a), ann_basetype(a), ann_footmarktype(a)

    def names(self):
        return sorted([cat_name(c) for c in self.categories])

    def occurring_names(self):
        return sorted(set([self.ann_name(a) for a in self.annotations]))

    def occurring_classes(self):
        return sorted(set([self.ann_class(a) for a in self.annotations]),
                      key=lambda c: c[0] + ' ' + c[1] + ' ' + c[2])


# Layout

def vert_span(a):
    return ann_y(a), ann_y(a) + ann_h(a)


def vert_spans_overlap(s1, s2):
    (l1, h1) = s1
    (l2, h2) = s2
    return h1 > l2 and h2 > l1


def overlapping_vert_span(spans, s):
    for i in range(len(spans)):
        if vert_spans_overlap(spans[i], s):
            return i
    return -1


def unify_vert_spans(s1, s2):
    (l1, h1) = s1
    (l2, h2) = s2
    return min([l1, l2]), max([h1, h2])


def divide_into_lines(ann):
    spans = []
    lines = defaultdict(list)
    ann = sorted(ann, key=lambda a: ann_x(a))
    for a in ann:
        span = vert_span(a)
        similar_index = overlapping_vert_span(spans, span)
        if similar_index >= 0:
            spans[similar_index] = unify_vert_spans(spans[similar_index], span)
        else:
            spans.append(span)
            similar_index = len(spans) - 1
        lines[similar_index].append(a)
    return [lines[i] for i in sorted(lines.keys(), key=lambda l: spans[l][0])]
