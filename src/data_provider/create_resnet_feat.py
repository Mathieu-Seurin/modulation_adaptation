from src.neural_toolbox.gpu_utils import FloatTensor, USE_CUDA
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

def precompute_image_feat(mode="train"):

    batchsize = 3

    path = os.path.join("/media/datas1/dataset/clevr/CLEVR_v1.0/images/", mode)
    path_to_feat = "/media/datas2/precomputed/clevr_res101_np"
    path_to_feat = os.path.join(path_to_feat, mode)

    if not os.path.exists(path_to_feat):
        os.mkdir(path_to_feat)

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

    path_list = []
    img_list = []

    for img_path in tqdm.tqdm(glob.glob(path+"/*.png")):

        # Compute path and name of the image
        img_name_raw = os.path.basename(img_path)
        img_name_raw_wo_ext, ext = os.path.splitext(img_name_raw)
        feat_path = os.path.join(path_to_feat, img_name_raw_wo_ext+".npy")

        # if os.path.exists(feat_path):
        #     continue

        # Load and transform image
        img = Image.open(img_path, mode="r")
        img = img.convert('RGB')
        img = transform(img).unsqueeze(0)

        # Store in list images and his path
        path_list.append(feat_path)
        img_list.append(img)

        assert len(path_list) == len(img_list)

        # if you reach batch limit, compute forward pass and store
        if len(path_list) == batchsize:

            batch = torch.cat(img_list, dim=0)
            batch = Variable(batch.type(FloatTensor), volatile=True)
            feats = model.forward(batch).data.cpu()

            # save each image at the right path
            for num_in_batch, img_save_path in enumerate(path_list):
                feat = feats[num_in_batch].numpy()
                np.save(img_save_path, feat)

            # clean after mess to redo
            path_list = []
            img_list = []

    # Because you can't exactly fall on good size of batches
    if len(path_list) > 0:
        assert len(path_list) == len(img_list)

        batch = torch.cat(img_list, dim=0)
        batch = Variable(batch.type(FloatTensor), volatile=True)
        feats = model.forward(batch).data.cpu()

        # save each image at the right path
        for num_in_batch, img_save_path in enumerate(path_list):
            feat = feats[num_in_batch].numpy()
            np.save(img_save_path, feat)


if __name__ == "__main__":
    precompute_image_feat("train")
    precompute_image_feat("val")

