# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Optional, Tuple


class FaceEmotionsDataset(Dataset):
    """Face Emotions dataset."""

    def __init__(self, csv_file, root_dir, classes, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.emotions_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.emotions_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get the path to the image
        img_name = os.path.join(self.root_dir, self.emotions_frame.iloc[idx, 0])
        # load the image as gray scaled image
        gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # get the emotion label
        emotion = self.emotions_frame.iloc[idx, 1].lower()
        # convert emotion string into id.
        emotion_id = self.classes.index(emotion)
        # define the sample
        sample = {"image": image, "emotion": emotion_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FER13(Dataset):
    """ `FER13 <https://www.kaggle.com/deadskull7/fer2013>` Dataset.
    Parameters
    ----------
    root: string
        Root directory of dataset where directory fer2013.csv exists.
    train: bool (optional)
        If True, creates dataset from training set, otherwise creates from
        test set.
    transform: callable (optional)
        A function/transform that takes in an PIL image and returns a
        transformed version.
    target_transform: callable (optional)
        A function/transform that takes in the target and transforms it.
    TODO
    ----
    download: bool (optional)
        If true, downloads the dataset from the internet and put it in
        root directory. If dataset is already download, it is not download
        again.
    """

    filename = "fer2013.csv"
    _repr_indent = 4
    classes = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # download: bool = False,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        dataframe = pd.read_csv(os.path.join(self.root, self.filename))

        if self.train:
            self.usage = "Training"
        else:
            self.usage = "PublicTest"

        self.data: Any = dataframe[
            dataframe["Usage"] == self.usage
        ].pixels.tolist()

        self.targets: Any = dataframe[
            dataframe["Usage"] == self.usage
        ].emotion.to_numpy(dtype=np.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Parameter
        ---------
        index: int

        Returns
        -------
        tuple: (image, target) where target is index of the target class.
        """
        pixels, target = self.data[index], self.targets[index]

        gray = np.array(
            list(map(int, pixels.split(" "))), dtype=np.uint8
        ).reshape((48, 48))
        color = np.repeat(gray[..., np.newaxis], 3, -1)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(color)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += [f"Split: {'Train' if self.train is True else 'Test'}"]
        if self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


if __name__ == "__main__":
    # DEV TESTS
    emotions = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    ]

    face_dataset = FaceEmotionsDataset(
        csv_file="csv/cleaned_data.csv", root_dir="img/", classes=emotions
    )

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample["image"].shape, sample["emotion"])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title("{}: {}".format(i, emotions[sample["emotion"]]))
        ax.axis("off")
        ax.imshow(sample["image"])

        if i == 3:
            plt.show()
            break
    """

    scale = tr.Rescale(64)
    fig = plt.figure()
    sample = face_dataset[-1]
    transformed_sample = scale(sample)

    plt.tight_layout()

    plt.imshow(transformed_sample['image'])
    plt.title('{}'.format(transformed_sample['emotion']))

    plt.show()
    """
