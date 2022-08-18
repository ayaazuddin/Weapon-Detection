import json
import glob
import torch
from IPython.display import Image  # for displaying images
import os
import random

from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

random.seed(108)


classes = []
input_dir = "./dataset/annotations/xmls"
output_dir = "./dataset/annotations2"
image_dir = "./dataset/images"


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


files = glob.glob(os.path.join(input_dir, '*.xml'))
# loop through each
for fil in files:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
        print(f"{filename} image does not exist!")
        continue

    result = []

    # parse the content of the xml file
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall('object'):
        label = obj.find("name").text
        # check for new classes and append to list
        if label not in classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")

    if result:
        # generate a YOLO format text file for each xml file
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))

# generate the classes file as reference
with open('classes.txt', 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))


# # Function to get the data from XML Annotation
# def extract_info_from_xml(xml_file):
#     root = ET.parse(xml_file).getroot()

#     # Initialise the info dict
#     info_dict = {}
#     info_dict['bboxes'] = []

#     # Parse the XML Tree
#     for elem in root:
#         # Get the file name
#         if elem.tag == "filename":
#             info_dict['filename'] = elem.text

#         # Get the image size
#         elif elem.tag == "size":
#             image_size = []
#             for subelem in elem:
#                 image_size.append(int(subelem.text))

#             info_dict['image_size'] = tuple(image_size)

#         # Get details of the bounding box
#         elif elem.tag == "object":
#             bbox = {}
#             for subelem in elem:
#                 if subelem.tag == "name":
#                     bbox["class"] = subelem.text

#                 elif subelem.tag == "bndbox":
#                     for subsubelem in subelem:
#                         bbox[subsubelem.tag] = int(subsubelem.text)
#             info_dict['bboxes'].append(bbox)

#     return info_dict


# # print(extract_info_from_xml('./dataset/annotations/xmls/ABbframe00154.xml'))

# # Dictionary that maps class names to IDs
# class_name_to_id_mapping = {"knife": 0,
#                             "billete": 1}

# # Convert the info dict to the required yolo format and write it to disk


# def convert_to_yolov5(info_dict):
#     print_buffer = []

#     # For each bounding box
#     for b in info_dict["bboxes"]:
#         try:
#             class_id = class_name_to_id_mapping[b["class"]]
#         except KeyError:
#             print("Invalid Class. Must be one from ",
#                   class_name_to_id_mapping.keys())

#         # Transform the bbox co-ordinates as per the format required by YOLO v5
#         b_center_x = (b["xmin"] + b["xmax"]) / 2
#         b_center_y = (b["ymin"] + b["ymax"]) / 2
#         b_width = (b["xmax"] - b["xmin"])
#         b_height = (b["ymax"] - b["ymin"])

#         # Normalise the co-ordinates by the dimensions of the image
#         image_w, image_h, image_c = info_dict["image_size"]
#         b_center_x /= image_w
#         b_center_y /= image_h
#         b_width /= image_w
#         b_height /= image_h

#         # Write the bbox details to the file
#         print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
#             class_id, b_center_x, b_center_y, b_width, b_height))

#     # Name of the file which we have to save
#     save_file_name = os.path.join(
#         "./dataset/annotations2", info_dict["filename"].replace("png", "txt"))

#     # Save the annotation to disk
#     print("\n".join(print_buffer), file=open(save_file_name, "w"))


# # Get the annotations
# annotations = [os.path.join('./dataset/annotations/xmls', x)
#                for x in os.listdir('./dataset/annotations/xmls') if x[-3:] == "xml"]
# annotations.sort()

# # Convert and save the annotations
# for ann in tqdm(annotations):
#     info_dict = extract_info_from_xml(ann)
#     convert_to_yolov5(info_dict)
# annotations = [os.path.join('/dataset/annotations2', x)
#                for x in os.listdir('./dataset/annotations2') if x[-3:] == "txt"]
