import os.path
import re

import unicodedata

from src.util.glyph_util import glyph_to_glyph


def strip_accents(s):
    """
    Author: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    :param s:
    :return:
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


input_data_dir = os.path.join("perseus", "data")
output_data_dir = os.path.join("dataset", "perseus.txt")
character_blacklist = "\W|\d|[a-zA-Z]|ʼ|_|ʽ|ʹ|²|½"

list_of_files = []
for root, dirs, files in os.walk(input_data_dir):
    for file in files:
        if file.endswith(".xml") and "grc" in file:
            list_of_files.append(os.path.join(root, file))

with open(output_data_dir, mode="w", encoding="UTF_8") as output:
    for xml_file in list_of_files:
        print(xml_file)
        try:
            with open(xml_file, mode="r", encoding='UTF_8') as f:
                full_text = f.read()
                full_text = re.sub(character_blacklist, "", full_text).upper()
                full_text = strip_accents(full_text)
                full_text = re.sub(character_blacklist, "", full_text)
                cleaned_text = [glyph_to_glyph(i) for i in full_text]
                output.write("".join(cleaned_text))
        except TypeError:
            continue
