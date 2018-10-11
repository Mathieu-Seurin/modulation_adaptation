from keras.datasets import fashion_mnist, cifar10
import numpy as np
from os import path

if __name__ == "__main__":

    path_to_precompute = 'precompute/'

    print("Computing mean and std for all datasets")

    # Fashion MNIST
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    np.save(path.join(path_to_precompute,"fashion_mean"), mean)
    np.save(path.join(path_to_precompute,"fashion_std"), std)

    # CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    mean = np.mean(x_train, axis=(0,1,2))
    print(mean)
    std = np.std(x_train, axis=(0,1,2))
    print(std)
    np.save(path.join(path_to_precompute,"cifar_mean"), mean)
    np.save(path.join(path_to_precompute,"cifar_std"), std)
