import json
from data_provider.dataset import CleveRDataset
from data_provider.tokenizer import CLEVRTokenizer
import numpy as np
import time

from PIL import Image
import matplotlib.pyplot as plt

from itertools import count

def tensor2image(tensor, num=0):

    image_np = tensor.permute(1,2,0).numpy()
    image_np[:,:,0] = image_np[:,:,0]*0.229 + 0.485
    image_np[:,:,1] = image_np[:,:,1]*0.224 + 0.456
    image_np[:,:,2] = image_np[:,:,2]*0.224 + 0.406

    plt.imshow(image_np)
    plt.savefig("/home/sequel/mseurin/test{}.png".format(num))

if __name__ == "__main__":

    dataset = CleveRDataset(mode="val")
    tokenizer = CLEVRTokenizer('/media/datas1/dataset/clevr/CLEVR_v1.0/dict.json')

    while True:

        i = np.random.randint(0, len(dataset))
        elem = dataset[i]
        print("============  {}  ==========".format(i))
        print(tokenizer.decode_question(elem['question']))
        print(elem['info']['question_raw'])

        print(tokenizer.decode_answer(elem['answer'][0]))
        print(elem['info']['answer_raw'])
        print("======================")

        tensor2image(elem['image'], num=i)

        time.sleep(10)





