import argparse
import logging
import tqdm

from models.models import ClfModel, count_good_prediction
from data_provider.dataset import SequentialTaskDataset
from models.gpu_utils import FloatTensor, LongTensor
from config import load_config_and_logger

import numpy as np
import torch
from torch.autograd import Variable

def full_train_test(config):

    # Create logger and init params
    logger = logging.getLogger()
    batch_size = config["model_params"]["batch_size"]
    n_epoch = config["env_params"]["n_epochs"]

    dataset = SequentialTaskDataset(config=config['env_params'],
                                    batch_size=batch_size)

    state_dim = {'vision_shape': dataset.example_shape, 'second_modality_shape': [1, dataset.total_number_task]}

    model = ClfModel(config=config["model_params"],
                     n_class=dataset.n_class,
                     state_dim=state_dim)


    accuracy_list_train = []
    accuracy_list_test = []

    for num_task in range(dataset.total_number_task):

        # Change the task
        dataset.new_task()
        model.new_task(num_task=num_task)

        for num_epoch in tqdm.tqdm(range(n_epoch)):

            loss = 0
            count_success = 0

            for num_batch, (batch, y) in enumerate(dataset.batch_gen()):

                # Convert all datas to torch.Tensor and Variable
                x,y = prepare_variables(batch, y, dataset.total_number_task, num_task)

                loss_temp, count_success_temp = model.optimize(x=x, y=y)
                loss += loss_temp
                count_success += count_success_temp

            num_batch += 1 # enumerate starts at 0, we have to deal with that

            logger.debug("\nloss : {}".format(float(loss.data)))
            accuracy = count_success / (num_batch * batch_size)
            logger.info("accuracy : {}".format(accuracy))

        logger.info("======= END OF TASK {} =======".format(num_task))
        test_score = test_single_task(model=model, dataset=dataset, task=num_task)
        accuracy_list_test.append(test_score)
        accuracy_list_train.append(accuracy)
        logger.info("Score for this task (test) : {:.2}% accuracy".format(test_score))


    mean_acc_across_task, accuracy_per_task = test_all_task(model, dataset)
    logger.info(str(accuracy_per_task))
    logger.info("Score for all task {:.2}% total accuracy".format(mean_acc_across_task))


def test_single_task(model, dataset, task):

    test_gen = dataset.batch_test_gen(task=task)
    n_test_sample = dataset.test_size

    count_success = 0
    for batch, labels_test in tqdm.tqdm(test_gen):
        x_test,y_test = prepare_variables(batch=batch, y=labels_test, total_number_task=dataset.total_number_task, num_task=task)

        yhat = model.forward(x_test)
        count_success += count_good_prediction(yhat=yhat, y=y_test)
    return count_success / n_test_sample

def test_all_task(model, dataset):

    n_test_sample = dataset.test_size
    total_test_size = dataset.total_test_size

    accuracy_per_task = []
    last_task = 0

    for num_task in range(dataset.n_task_done):
        accuracy_per_task.append(0)
        for batch, labels_test in dataset.batch_test_gen(task=num_task):
            x_test,y_test = prepare_variables(batch=batch, y=labels_test, total_number_task=dataset.total_number_task, num_task=num_task)
            yhat = model.forward(x_test)

            accuracy_per_task[num_task] += count_good_prediction(yhat=yhat, y=y_test)

        # todo : don't assume that all test size are the same
        accuracy_per_task[num_task] /= n_test_sample
        logger.info("Task {} : {:.2}% accuracy".format(num_task, accuracy_per_task[num_task]))

    return np.sum(accuracy_per_task) / dataset.n_task_done, accuracy_per_task


def prepare_variables(batch, y, total_number_task, num_task):
    """
    Prepare all torch.Tensor, cuda, and Variable
    """

    batch = FloatTensor(batch)
    y = Variable(LongTensor(y))
    task = torch.zeros(batch.size(0), total_number_task).type(FloatTensor)
    task[:, num_task] = 1

    x = {'vision_shape': Variable(batch), 'second_modality_shape': Variable(task)}

    return x,y



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-exp_dir", type=str, default="out", help="Directory all results")
    parser.add_argument("-env_config", type=str, help="Which file correspond to the experiment you want to launch ?")
    parser.add_argument("-model_config", type=str, help="Which file correspond to the experiment you want to launch ?")
    parser.add_argument("-model_extension", type=str, help="Do you want to override parameters in the model file ?")
    parser.add_argument("-display", type=str, help="Display images or not")
    parser.add_argument("-seed", type=int, default=0, help="Manually set seed when launching exp")
    parser.add_argument("-device", type=int, default=-1, help="Manually set GPU")

    args = parser.parse_args()

    config, exp_identifier, save_path = load_config_and_logger(env_config_file=args.env_config,
                                                               model_config_file=args.model_config,
                                                               model_ext_file=args.model_extension,
                                                               exp_dir=args.exp_dir,
                                                               seed=args.seed)

    logger = logging.getLogger()

    device = args.device
    if device != -1:
        torch.cuda.set_device(device)
        logger.info("Using device {}".format(torch.cuda.current_device()))
    else:
        logger.info("Using default device from env")

    full_train_test(config=config)