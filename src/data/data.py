import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision.transforms import Pad
from torch.utils.data import DataLoader

import numpy as np


class WhiteSpacePad:
    def __call__(self, pic):
        pic_size = pic.size
        max_size = max(pic.size)
        horiz_pad, ver_pad = tuple(map(lambda size: (max_size - size) / 2,
                                       pic_size))
        padding = (int(np.floor(horiz_pad)), int(np.floor(ver_pad)),
                   int(np.ceil(horiz_pad)), int(np.ceil(ver_pad)))
        pad = Pad(padding, fill=255, padding_mode='constant')
        return pad(pic)
    

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size, batch_size):
        super().__init__()
          
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
          
        self.transform = transforms.Compose([WhiteSpacePad(),
                                            transforms.Resize(self.img_size),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
  
    def setup(self, stage=None):
             
        self.train_data = datasets.ImageFolder(self.data_dir, 
                                               transform=self.transform)
  
    def train_dataloader(self):
        
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size,
                          shuffle=True)
    



