import torch
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image
import os

class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode, type_data):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.type = type_data
        self.preprocess()
        
        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            label_path = os.path.join(self.label_path, str(i)+'.png')
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])
            
    def __getitem__(self, index):
        
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        if self.type == "both":
            image = Image.open(img_path)
            label = Image.open(label_path)
            return self.transform_img(image), self.transform_label(label)
        elif self.type == "image":
            image = Image.open(img_path)
            return self.transform_img(image), None
        elif self.type == "label":
            label = Image.open(label_path)
            return None, self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode, type_data="both", gray=False):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode
        self.gray = gray
        self.type = type_data

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(128))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if self.gray:
            options.append(transforms.Grayscale(1))
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(128))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize), interpolation=Image.NEAREST))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform
        
    def loader(self):
        transform_img = self.transform_img(True, True, False, False) 
        transform_label = self.transform_label(True, True, False, False)  
        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode, self.type)
        self.dataset = dataset

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                            #  shuffle=True,
                                             shuffle=self.mode,
                                             num_workers=2,
                                             drop_last=False)
        return loader
