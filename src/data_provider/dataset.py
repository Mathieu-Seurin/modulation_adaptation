import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist, cifar10
import torch
import h5py

from skimage import io
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import os
from PIL import Image

class ImageLoader(object):
    def __init__(self):
        raise NotImplementedError("Later")

class Shuffler(object):
    def __init__(self, n_shuffle, in_size):
        self.n_shuffle = n_shuffle

        self.shuffler = np.arange(in_size)
        np.random.shuffle(self.shuffler)

    def shuffle(self, x):

        assert isinstance(x, np.ndarray)
        assert x.ndim >= 3, "X need to be of dim 3, your dim is {}, forgot batch dim ?".format(x.ndim)

        if x.ndim > 3:
            raise NotImplementedError("Can only deal with mnist style dataset")
            image_shape = x.shape
        else:
            image_shape = np.expand_dims(x,1).shape # Add features dimension for mnist style dataset

        # vectorize example, switch pixel
        x = x.reshape(image_shape[0], -1)
        x = x[:,self.shuffler]
        x = x.reshape(*image_shape)
        return x

    def __call__(self, x):
        return self.shuffle(x)

class SequentialTaskDataset(object):
    def __init__(self, config, batch_size):

        self.batch_size = batch_size
        self.n_pixel_change = config["n_pixel_change"]
        self.total_number_task = config["n_task"]
        dataset = config["dataset"]

        if dataset=="fashion_mnist":
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
            self.mean = np.expand_dims(np.load("precompute/fashion_mean.npy"), 0)
            self.std = np.expand_dims(np.load("precompute/fashion_mean.npy"), 0)
            self.n_class = 10

        elif dataset=="cifar10":
            raise NotImplementedError("Not yet")
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
            self.n_class = 10

        else:
            raise NotImplementedError("No way.")

        self.example_size = self.x_train[0].size

        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

        self.all_shuffler = []


        # compute the shape of the image that is going to be fed to the model (need to add batch dim)
        temp_shuffler = Shuffler(n_shuffle=self.n_pixel_change,
                                 in_size=self.example_size)

        self.example_shape = temp_shuffler(np.expand_dims(self.x_train[0],0)).shape


    def normalize(self, x):
        """
        Move this to Image Loader
        """
        return (x - self.mean) / self.std


    def batch_gen(self):

        n_ite = self.train_size//self.batch_size

        for num_batch in range(n_ite):
            random_select = np.random.choice(np.arange(self.x_train.shape[0]),size=self.batch_size, replace=True)
            batch = self.normalize(self.shuffler(self.x_train[random_select]))
            labels = self.y_train[random_select]
            yield batch, labels

    def batch_test_gen(self, task=None):

        if task is None:
            shuffler = self.shuffler
        else:
            assert isinstance(task, int), "task need to be index number is : {} {}".format(type(task), task)
            shuffler = self.all_shuffler[task]

        for num_ite, id_begin in enumerate(range(0, self.test_size, self.batch_size)):
            slice_selected = slice(id_begin, min(id_begin+self.batch_size, self.test_size)) # don't worry about bound, slice is sweet
            batch_test = self.normalize(shuffler(self.x_test[slice_selected]))
            labels = self.y_test[slice_selected]
            yield batch_test, labels


    def new_task(self):

        self.all_shuffler.append(Shuffler(n_shuffle=self.n_pixel_change,
                                          in_size=self.example_size))
        self.shuffler = self.all_shuffler[-1]


    def __len__(self):
        return self.train_size

    @property
    def total_test_size(self):
        return self.test_size * len(self.all_shuffler)

    @property
    def n_task_done(self):
        return len(self.all_shuffler)

class ImageClassifDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='fashion-mnist', mode="train", shuffle_label=False):

        self.train_val_ratio = 0.8

        if dataset== 'fashion-mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=(0.1307,),
                          std=(0.3081,))
            ])

            self.example_shape = (1,1,28,28)
            self.n_class = 10

            x_train = np.expand_dims(x_train, axis=3)
            x_test = np.expand_dims(x_test, axis=3)

        elif dataset== 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=(0.4914, 0.4822, 0.4465),
                          std=(0.247, 0.243, 0.261))
            ])

            self.example_shape = (1,3,32,32)
            self.n_class = 10

            y_train = np.squeeze(y_train, axis=1)
            y_test = np.squeeze(y_test, axis=1)

        else:
            NotImplementedError("No other dataset supported, you asked for : {}".format(dataset))

        self.dataset = {}
        if mode == "train":
            self.dataset['x'] = x_train[:int(self.train_val_ratio*len(x_train))]
            self.dataset['y'] = y_train[:int(self.train_val_ratio*len(x_train))]
        elif mode=="val":
            self.dataset['x'] = x_train[int(self.train_val_ratio*len(x_train)):]
            self.dataset['y'] = y_train[int(self.train_val_ratio*len(x_train)):] # check shuffled dataset
        else:
            self.dataset['x'] = x_test
            self.dataset['y'] = y_test

        self.question_shape = [1]
        if shuffle_label :

            label_switch = [6, 3, 4, 5, 9, 8, 0, 1, 7, 2] # Arbitrary shuffle, osef

            new_labels = self.dataset['y'].copy()

            for i in range(self.n_class):
                new_labels[np.where(self.dataset['y']==i)] = label_switch[i]

            self.dataset['y'] = new_labels


            self.task = np.array([1])
            pass
        else:
            self.task = np.array([0])

        #self.dataset['y'] = tf.keras.utils.to_categorical(self.dataset['y'])

    def __getitem__(self, item):
        return {'image' : self.transform(self.dataset['x'][item]), 'y' : self.dataset['y'][item], 'task': self.task}

    def __len__(self):
        return len(self.dataset['x'])


class CleveRDataset(torch.utils.data.Dataset):
    def __init__(self, mode, images_type="normalized", debug=False):

        self._mode = mode
        self.debug=debug

        self.images_type = images_type

        self.raw_img_path = '/media/datas1/dataset/clevr/CLEVR_v1.0/images/{}/'.format(mode)
        self.precomp_feat_path = '/media/datas2/precomputed/clevr_res101/{}/'.format(mode)

        self.n_class = 30
        self.vocab_size = 85

        self._len = 0

        self.transform = Compose([
            Resize([224,224]),
            ToTensor(),
            Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                      std=torch.FloatTensor([0.229,0.224,0.224]))
        ])

        self.example_shape = self[0]['image'].unsqueeze(0).size()
        assert len(self.example_shape) == 4, "Wrong number of dimension, should be 4. Number of dimension is {}".format(len(self.example_shape))
        self.question_shape = self[0]['question'].unsqueeze(0).size()

    def __len__(self):
        with h5py.File('/media/datas1/dataset/clevr/CLEVR_v1.0/{}_questions.h5'.format(self._mode), 'r') as info_supp:
            return len(info_supp['answers'])

    def __getitem__(self, item):

        with h5py.File('/media/datas1/dataset/clevr/CLEVR_v1.0/{}_questions.h5'.format(self._mode), 'r') as info_supp:
            img_name = info_supp['image_filenames'][item].decode("UTF-8")
            question = info_supp['questions'][item]
            answer = np.array([info_supp['answers'][item]])
            orig_idx = np.array([info_supp['orig_idxs'][item]])

            if self.debug:
                question_raw = info_supp['questions_raw'][item]
                answer_raw = info_supp['answers_raw'][item]

            self._len = len(info_supp['answers'])

        assert answer[0] < 32, "answer is {}".format(answer)

        if self.images_type == "pretrained":

            with h5py.File("/home/sequel/mseurin/precomputed/clevr_res101/{}_images.h5".format(self._mode),'r') as images:
                image_name_wo_extension, extension = os.path.splitext(img_name)
                index = int(image_name_wo_extension.split('_')[2])

                img = torch.from_numpy(images['images'][index])

        elif self.images_type == "normalized":

            img_path = os.path.join(self.raw_img_path, img_name)

            # image and transform
            img = Image.open(img_path, mode="r")
            img = img.convert('RGB')
            img = self.transform(img)
        else:
            raise NotImplementedError("images type {} cannot be done.".format(self.images_type))

        #other additionnal info
        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)

        if self.debug:
            info = {'img_path': img_path, 'q_id' : orig_idx, 'question_raw': question_raw, 'answer_raw': answer_raw}
            sample = {'image': img, 'question': question, 'answer': answer, 'info': info}
        else:
            sample = {'image': img, 'question': question, 'answer': answer}

        return sample

