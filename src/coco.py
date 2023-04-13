import json
from collections import defaultdict


# Attributes of annotations:
# * id
# * image_id
# * category_id
# * bbox ( 4 integers )
# * area ( width * height )
# * iscrowd
# * tags [ BaseType and FootMarkType ]
# * seg_id


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
        return a['tags']['FootMarkType'][0]
    else:
        return ''


# Attributes of categories:
# * id
# * name
# * supercategory: "Greek"
def cat_id(c):
    return c['id']


def cat_name(c):
    return c['name']


# Attributes of images:
# * id
# * bln_id ( same as id )
# * file_name
# * img_url ( same as file_name )
# * width
# * height
# * date_captured
# * license
def img_id(i):
    return i['id']


def img_file(i):
    return i['file_name']


# Attributes of licenses:
# * id
# * name
# * url

class COCOread:
    def __init__(self, coco):
        with open(coco, encoding="utf-8") as f:
            info = json.load(f)
        self.annotations = info['annotations']
        self.categories = info['categories']
        self.images = info['images']
        self.licenses = info['licenses']

        self.id_to_ann = {}
        for a in self.annotations:
            self.id_to_ann[ann_id(a)] = a

        self.id_to_cat = {}
        for c in self.categories:
            self.id_to_cat[cat_id(c)] = c

        self.id_to_img = {}
        for i in self.images:
            self.id_to_img[img_id(i)] = i

        self.img_to_anns = defaultdict(list)
        for a in self.annotations:
            self.img_to_anns[ann_img_id(a)].append(a)

        self.name_to_anns = defaultdict(list)
        for a in self.annotations:
            id = ann_cat_id(a)
            cat = self.id_to_cat[id]
            name = cat_name(cat)
            self.name_to_anns[name].append(a)

        self.name_to_id = {}
        for c in self.categories:
            self.name_to_id[cat_name(c)] = cat_id(c)

    def img_ids(self):
        return sorted(self.id_to_img.keys())

    def ann_name(self, a):
        return cat_name(self.id_to_cat[ann_cat_id(a)])

    def ann_class(self, a):
        return (self.ann_name(a), ann_basetype(a), ann_footmarktype(a))

    def names(self):
        return sorted([cat_name(c) for c in self.categories])

    def occurring_names(self):
        return sorted(set([self.ann_name(a) for a in self.annotations]))

    def occurring_classes(self):
        return sorted(set([self.ann_class(a) for a in self.annotations]), \
                      key=lambda c: c[0] + ' ' + c[1] + ' ' + c[2])


class COCO:
    def __init__(self, categories, licenses):
        self.info = {}
        self.info['categories'] = categories
        self.info['licenses'] = licenses
        self.info['images'] = []
        self.info['annotations'] = []

    def add_image(self, img):
        self.info['images'].append(img)

    def add_annotation(self, ann):
        self.info['annotations'].append(ann)

    def write(self, coco):
        with open(coco, 'w', encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=4)
