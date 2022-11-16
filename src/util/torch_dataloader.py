import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.classification.alexnet import alex_init
from src.util import glyph_util
from src.util.dir_util import get_input_img_paths


class VectorLoader(Dataset):
    def __init__(self, language_file, annotations_file, img_dir, transform=None, target_transform=None):
        with open(language_file, mode="r", encoding="UTF-8") as lang_file:
            self.language_text = lang_file.read()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_vectors, self.vector_labels = alex_init(img_dir)
        self.vector_size = len(self.img_vectors[0])
        self.transform = transform
        self.target_transform = target_transform

        vectors = {}
        for v, l in zip(self.img_vectors, self.vector_labels):
            if l not in vectors:
                vectors[l] = [torch.from_numpy(np.array(v)).float()]
            else:
                vectors[l].append(torch.from_numpy(np.array(v)).float())
        self.img_vectors = vectors

        random.seed(0)

    def get_vector_size(self):
        return self.vector_size

    def __len__(self):
        return len(self.language_text)

    def __getitem__(self, index):
        label = self.language_text[index]
        if glyph_util.glyph_class_to_name(label) is None:
            print(label)
        vector = random.choice(self.img_vectors[glyph_util.glyph_class_to_name(label)])
        if self.transform:
            vector = self.transform(vector)
        if self.target_transform:
            label = self.target_transform(label)
        return vector, glyph_util.glyph_to_index(label)


class ImageLoader(Dataset):
    def __init__(self, language_file, annotations_file, img_dir, transform=None, target_transform=None):
        with open(language_file, mode="r", encoding="UTF-8") as lang_file:
            self.language_text = lang_file.read()
        self.img_labels = pd.read_csv(annotations_file)
        self.imgs = get_input_img_paths(img_dir, by_dir=True)
        self.transform = transform
        print("transform", self.transform)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.language_text)

    @staticmethod
    def get_vector_size():
        return None

    def __getitem__(self, index):
        label = self.language_text[index]
        if glyph_util.glyph_class_to_name(label) is None:
            print(label)
        path = random.choice(self.imgs[glyph_util.glyph_class_to_name(label)])
        input_tensor = Image.open(path)
        if self.transform:
            input_tensor = self.transform(input_tensor)
            input_tensor.unsqueeze(0)
        if self.target_transform:
            label = self.target_transform(label)
        return input_tensor, glyph_util.glyph_to_index(label)
