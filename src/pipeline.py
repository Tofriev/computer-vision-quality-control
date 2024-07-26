from src.config import Config
from src.data_preprocessing.dataset import MyDataset
from glob import glob
from sklearn.model_selection import train_test_split
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image


class Pipeline:
    def __init__(self, both_box_types=False, box_type="boxtype1"):
        self.both_box_types = both_box_types
        self.box_type = box_type
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None
        self.data_dir = Config.DATA_DIR
        self.test_size = Config.TEST_SIZE
        self.random_state = Config.RANDOM_STATE

    def train_val_test_split(self):
        """
        Split the dataset into training, validation, and testing sets.
        """
        if self.both_box_types:
            box_types = ["boxtype1", "boxtype2"]
        else:
            box_types = [self.box_type]

        images = []
        labels = []
        for box_type in box_types:
            images_dir = os.path.join(self.data_dir, "raw", box_type)
            print(images_dir)
            # sort lists in order to allocate images as lables into the same bucket when spliting
            images.extend(sorted(glob(os.path.join(images_dir, "*.jpg"))))
            labels.extend(sorted(glob(os.path.join(images_dir, "*.xml"))))
            img = Image.open(images[0])
            plt.imshow(img)
            plt.show()
            print(labels)

        train_data, test_data, labels_train, labels_test = train_test_split(
            images, labels, test_size=self.test_size, random_state=self.random_state
        )

        train_data, val_data, labels_train, labels_val = train_test_split(
            train_data,
            labels_train,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.labels_train = labels_train
        self.labels_val = labels_val
        self.labels_test = labels_test

    def create_datasets(self, width, height, classes):
        """
        Create training, validation, and testing datasets.
        """
        self.train_dataset = MyDataset(
            self.train_data, width, height, classes, self.labels_train
        )
        self.val_dataset = MyDataset(
            self.val_data, width, height, classes, self.labels_val
        )
        self.test_dataset = MyDataset(
            self.test_data, width, height, classes, self.labels_test
        )

    def run(self):
        self.train_val_test_split()
        self.create_datasets(Config.RESIZE_TO, Config.RESIZE_TO, Config.CLASSES)
        self.train_dataset.len()
