import csv
import fnmatch
import json
import os
import re
import shutil

from coco import COCOread, COCO


def path_to_linux(path):
    return path.replace('\\', '/')


def find_images(root):
    paths = []
    for dir, _, files in os.walk(root):
        for filename in files:
            if fnmatch.fnmatch(filename.lower(), '*.jpg') or \
                    fnmatch.fnmatch(filename.lower(), '*.png'):
                path = os.path.join(dir, filename)
                rel_path = path_to_linux(os.path.relpath(path, root))
                paths.append(rel_path)
    return paths


def read_transcriptions_csv(trans_file):
    with open(trans_file, encoding="utf-8") as f:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(f)]


def txt2int(s):
    match = re.match(r'.*/txt([0-9]+)/.*', s)
    if match:
        return int(match.group(1))
    else:
        return 0


def annotation_record(trans, id, cat_id, image_id):
    record = {}
    x = int(trans['min_x'])
    y = int(trans['min_y'])
    w = int(trans['max_x']) - x
    h = int(trans['max_y']) - y
    record['area'] = w * h
    record['bbox'] = [x, y, w, h]
    record['category_id'] = cat_id
    record['id'] = id
    record['image_id'] = image_id
    record['iscrowd'] = 0
    record['seg_id'] = id
    record['tags'] = {'BaseType': ['bt1']}
    record['score'] = float(trans['certainty'])
    return record


if __name__ == '__main__':
    dir_train = '../HomerCompTraining'
    dir_test = '../HomerCompTesting'
    trans_file_csv = '../output/transcriptions_800_200_no_inner_window.csv'
    template_coco_file = '../output/template.json'
    coco_train = COCOread(os.path.join(dir_train, 'HomerCompTrainingReadCoco.json'))
    coco_test = COCO(coco_train.categories, coco_train.licenses)
    images = sorted(find_images(dir_test), key=lambda s: (txt2int(s), s))

    with open(template_coco_file) as template:
        template_dict = json.loads(template.read())

    path_to_img_id = {}
    for record in template_dict["images"]:
        for path in images:
            if record["file_name"] == './' + path:
                img_record = record
                img_id = img_record["id"]
                path_to_img_id[path] = img_id
                coco_test.add_image(img_record)

    transcriptions = read_transcriptions_csv(trans_file_csv)
    for i, trans in enumerate(transcriptions):
        cat_id = coco_train.name_to_id[trans['glyph']]
        image_id = path_to_img_id[path_to_linux(trans['image_path'])]
        ann_record = annotation_record(trans, i, cat_id, image_id)
        coco_test.add_annotation(ann_record)
    coco_test.write(os.path.join(dir_test, 'output', 'HomerCompTestingCoco.json'))
    shutil.make_archive(os.path.join(dir_test, 'HomerCompTestingCoco'), 'zip',
                        os.path.join(dir_test, 'output'))
