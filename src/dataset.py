import os
import torch
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import transform as tr

class FaceEmotionsDataset(Dataset):
    """Face Emotions dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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
    
    def __len__(self):
        return(len(self.emotions_frame))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.emotions_frame.iloc[idx, 0])
        image = io.imread(img_name, as_gray=True)
        emotion = self.emotions_frame.iloc[idx, 1]
        sample = {'image': image, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)
        
        return(sample)


if __name__ == "__main__":
    # DEV TESTS
    
    face_dataset = FaceEmotionsDataset(csv_file='csv/cleaned_data.csv',
                                       root_dir='img/')
    
    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        
        print(i, sample['image'].shape, sample['emotion'])

        ax = plt.subplot(1, 4, i+1)
        plt.tight_layout()
        ax.set_title('{}: {}'.format(i, sample['emotion']))
        ax.axis('off')
        ax.imshow(sample['image'])

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