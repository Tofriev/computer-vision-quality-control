from torch.utils.data import Dataset
import glob as glob
import os
import cv2
from xml.etree import ElementTree as et
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, image_paths, labels, width, height, classes, transforms=None):
        self.transforms = transforms
        self.image_paths = sorted(image_paths)
        self.labels = sorted(labels)
        self.height = height
        self.width = width
        self.classes = classes

    def __getitem__(self, idx):
        # image path
        image_path = self.image_paths[idx]
        # read image
        image = cv2.imread(image_path)
        # convert color format and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        # get xml file for the image to calculate IoU later on
        annot_file_path = self.labels[idx]

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall("object"):
            # get the coordinates of the bounding box in xml file
            if member.find("name").text == "CamFront":
                labels.append(self.classes.index(member.find("name").text))
                xmin = int(member.find("bndbox").find("xmin").text)
                xmax = int(member.find("bndbox").find("xmax").text)
                ymin = int(member.find("bndbox").find("ymin").text)
                ymax = int(member.find("bndbox").find("ymax").text)

                # resize bounding box
                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                ymax_final = (ymax / image_height) * self.height

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                # bounding box to tensor
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # calculate the area of the bounding boxes
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                labels = torch.as_tensor(labels, dtype=torch.int64)
                # make a target dict
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["area"] = area

                image_id = torch.tensor([idx])
                target["image_id"] = image_id

                # apply transformations of bounding box
                if self.transforms:
                    sample = self.transforms(
                        image=image_resized, bboxes=target["boxes"], labels=labels
                    )
                    image_resized = sample["image"]
                    target["boxes"] = torch.Tensor(sample["bboxes"])

        return image_resized, target

    def __len__(self):
        return len(self.image_paths)
