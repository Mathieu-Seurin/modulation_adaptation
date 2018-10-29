from src.models.gpu_utils import FloatTensor, USE_CUDA
import torch
from torch.autograd import Variable

if USE_CUDA:
    torch.cuda.set_device(0)

import numpy as np

from src.data_provider.dataset import CleveRDataset

from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from PIL import Image
import os
import glob
import tqdm

from torch import nn

import h5py

def precompute_image_feat(mode="train"):

    batchsize = 100

    path = os.path.join("/media/datas1/dataset/clevr/CLEVR_v1.0/images/", mode)
    path_to_feat = "/media/datas2/precomputed/clevr_res101/"

    if not os.path.exists(path_to_feat):
        os.mkdir(path_to_feat)

    h5_location = os.path.join(path_to_feat, "{}_images.h5".format(mode))

    images_gen = glob.glob(path + "/*.png")
    images_gen.sort()

    f = h5py.File(h5_location, "w")
    img_filename = f.create_dataset("images",shape=(len(images_gen),1024,14,14))

    model = models.resnet101(pretrained=True)

    modules = list(model.children())[:-3]
    model = nn.Sequential(*modules)

    model = model.eval()


    if USE_CUDA:
        model = model.cuda()

    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])

    transform = Compose([
        Resize([224, 224]),
        ToTensor(),
        Normalize(mean=mean,
                  std=std)
    ])

    img_list = []
    last_index = 0

    for img_num, img_path in tqdm.tqdm(enumerate(images_gen), total=len(images_gen)):

        # Compute name of image to place it at the good spot
        img_name_raw = os.path.basename(img_path)
        img_name_raw_wo_ext, ext = os.path.splitext(img_name_raw)

        index = int(img_name_raw_wo_ext.split('_')[2])

        assert img_num == index

        # Load and transform image
        img = Image.open(img_path, mode="r")
        img = img.convert('RGB')
        img = transform(img).unsqueeze(0)

        # Store in list images and his path
        img_list.append(img)

        # if you reach batch limit, compute forward pass and store
        if len(img_list) == batchsize:

            batch = torch.cat(img_list, dim=0)
            batch = Variable(batch.type(FloatTensor), volatile=True)
            feats = model.forward(batch).data.cpu().numpy()

            img_filename[last_index:last_index+batchsize] = feats

            # clean after mess to redo
            img_list = []
            last_index += batchsize

    # Because you can't exactly fall on good size of batches
    if len(img_list) > 0:

        batch = torch.cat(img_list, dim=0)
        batch = Variable(batch.type(FloatTensor), volatile=True)
        feats = model.forward(batch).data.cpu()

        # save each image at the right path
        img_filename[last_index:last_index + batchsize] = feats

if __name__ == "__main__":
    precompute_image_feat("train")
    precompute_image_feat("val")

