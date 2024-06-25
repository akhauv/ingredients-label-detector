import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image

class LabelDetectionDataset(Dataset):
    def __init__(self, image_path, annotation_path, transforms=None):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_path) if f.endswith('.jpeg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        annotation_file = os.path.join(self.annotation_dir, image_file.replace('.jpeg', '.xml'))
        image = Image.open(os.path.join(self.image_dir, image_file)).convert("RGB")

        boxes, labels = self._parse_annotation(annotation_file)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def _parse_annotation(self, annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.append(label)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes, labels
