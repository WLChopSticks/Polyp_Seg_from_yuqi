from __future__ import print_function, division
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
from mytransformation_2inputs import ToTensor

class Dataset_unet(Dataset):

    def __init__(self, image_dir, label_dir, boundary_dir=None, transform=None):
        self.image_ids = os.listdir(image_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        if boundary_dir is not None: self.boundary_dir = boundary_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx, with_boundary=False):
        if not with_boundary:
            sample = self.getDataDict(idx)
        else:
            sample = self.getDataDictWithBoundary(idx)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def getDataDict(self, idx):
        image_name = self.image_ids[idx]
        label_name = self.image_ids[idx]
        if 'cvc-612' in label_name:
            label_name = label_name.split('.')[0] + '.tif'

        image = Image.open(self.image_dir + image_name).convert('RGB')
        label = Image.open(self.label_dir + label_name).convert('L')

        # return dictionary
        return {'image': image, 'label': label, 'num': image_name}


    def getDataDictWithBoundary(self, idx):
        image_name = self.image_ids[idx]
        label_name = self.image_ids[idx]
        boundary_name = self.image_ids[idx]
        if 'cvc-612' in label_name:
            label_name = label_name.split('.')[0] + '.tif'

        image = Image.open(self.image_dir + image_name).convert('RGB')
        label = Image.open(self.label_dir + label_name).convert('L')
        boundary = Image.open(self.boundary_dir + boundary_name).convert('L')

        # return dictionary
        return {'image': image, 'label': label, 'boundary': boundary, 'num': image_name}


if __name__ == '__main__':
    image_dir = '../train/images/'
    label_dir = '../train/labels/'

    transform1 = transforms.Compose([ToTensor()])
    dataset = Dataset_unet(image_dir, label_dir, transform=transform1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
        print('i_batch:{}'.format(i_batch))

