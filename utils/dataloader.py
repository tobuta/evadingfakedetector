import cv2
from torch.utils.data import Dataset
import os
from PIL import Image, ImageFilter



class MyDataSet(Dataset):

    def __init__(self, dataPath, images_class, transform=None):
        self.dataPath = dataPath
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return (len(os.listdir(self.dataPath)))

    def __getitem__(self, item):

        ImageList = os.listdir(self.dataPath)
        ImagePath = os.path.join(self.dataPath, ImageList[item])
        img = cv2.imread(ImagePath)
        blur = cv2.GaussianBlur(img, (5, 5), 0.6, 0.6)
        img1 = Image.fromarray(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
        label = self.images_class
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, label
