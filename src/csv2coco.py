import csv
import os
import re
import fnmatch
from PIL import Image

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
	with open(trans_file) as f:
		return [{ k: v for k, v in row.items() } for row in csv.DictReader(f)]

def txt2int(s):
	match = re.match(r'.*/txt([0-9]+)/.*', s)
	if match:
		return int(match.group(1))
	else:
		return 0

def image_record(root, path, id):
	record = {}
	img = Image.open(os.path.join(root, path))
	(w, h) = img.size
	record['id'] = id
	record['bln_id'] = id
	record['file_name'] = './' + path
	record['img_url'] = './' + path
	record['width'] = w
	record['height'] = h
	record['date_captured'] = None
	record['license'] = 9
	return record

def annotation_record(trans, id, cat_id, image_id):
	record = {}
	x = int(trans['min_x'])
	y = int(trans['min_y'])
	w = int(trans['max_x']) - x
	h = int(trans['max_y']) - y
	record['area'] = w * h
	record['bbox'] = [ x, y, w, h ]
	record['category_id'] = cat_id
	record['id'] = id
	record['image_id'] = image_id
	record['iscrowd'] = 0
	record['seg_id'] = id
	record['tags'] = { 'BaseType': ['bt1'] }
	return record

if __name__ == '__main__':
	dir_train = '/home/mjn/Data/ICDAR2023/HomerCompTraining'
	dir_test = '/home/mjn/Data/ICFHR2022/HomerCompTesting'
	trans_file_csv = 'transcriptions.csv'
	coco_train = COCOread(os.path.join(dir_train, 'HomerCompTrainingReadCoco.json'))
	coco_test = COCO(coco_train.categories, coco_train.licenses)
	images = sorted(find_images(dir_test), key=lambda s: (txt2int(s), s))
	path_to_img_id = {}
	for path, id in zip(images, range(len(images))):
		img_record = image_record(dir_test, path, id)
		coco_test.add_image(img_record)
		path_to_img_id[path] = id
	transs = read_transcriptions_csv(trans_file_csv)
	for trans, id in zip(transs, range(len(transs))):
		cat_id = coco_train.name_to_id[trans['glyph']]
		image_id = path_to_img_id[path_to_linux(trans['image_path'])]
		ann_record = annotation_record(trans, id, cat_id, image_id)
		coco_test.add_annotation(ann_record)
	coco_test.write(os.path.join(dir_test, 'HomerCompTestingCoco.json'))
