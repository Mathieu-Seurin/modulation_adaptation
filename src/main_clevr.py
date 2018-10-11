import argparse
import logging
import tqdm

import torch

from models.models import ClfModel, count_good_prediction
from data_provider.dataset import CleveRDataset
from models.gpu_utils import FloatTensor, LongTensor
from config import load_config_and_logger

import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader

def full_train_test(config, debug=False):

    # Create logger and init params
    logger = logging.getLogger()
    batch_size = config["model_params"]["batch_size"]
    n_epoch = config["env_params"]["n_epochs"]

    dataset = CleveRDataset(mode="train")

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8)

    input_info = {'vision_shape': dataset.example_shape,
                  'second_modality_shape': dataset.question_shape,
                  "vocab_size" : dataset.vocab_size,
                  "second_modality_type": "text"}

    model = ClfModel(config=config["model_params"],
                     n_class=dataset.n_class,
                     input_info=input_info)

    accuracy_list_train = []
    accuracy_list_val = []

    for num_epoch in range(n_epoch):

        model.train()
        logging.info("Epoch #{} begin :".format(num_epoch))
        loss = 0
        count_success = 0

        if debug:
            sample_generator = enumerate(dataloader)
        else:
            sample_generator = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

        for num_batch, batch in sample_generator:
            if debug and num_batch > 50:
                break

            # Convert all datas to torch.Tensor and Variable
            x,y = prepare_variables(batch)

            loss_temp, yhat = model.optimize(x=x, y=y)
            count_success_temp = count_good_prediction(yhat=yhat, y=y)
            loss += loss_temp
            count_success += count_success_temp

            loss_temp, x, y, yhat = None, None, None, None

        # Train score
        logger.debug("\nloss : {}".format(float(loss)))

        # ALTERNATIVE TRAIN SCORE FOR TEST
        #train_score = count_success / len(dataset)
        surrog_len_dataset = (num_batch + 1) * batch_size
        train_score_estim = count_success / surrog_len_dataset
        train_score_exact = count_success / len(dataset)

        logger.info("Accuracy train (Estimate) : {:.2}".format(train_score_estim))
        logger.info("Accuracy train (Exact) : {:.2}".format(train_score_exact))
        accuracy_list_train.append(train_score_exact)

        # Validation score
        val_score = test_model(model=model, dataset_mode="val", batch_size=batch_size, debug=debug)
        logger.info("Accuracy val (exact) : {:.2}".format(val_score))
        accuracy_list_val.append(val_score)

    # Test score
    # test_score = test_model(model=model, dataset_mode="test", batch_size=batch_size)
    # logger.info("Accuracy test : {:.2}% accuracy".format(test_score))


def test_model(model, dataset_mode, batch_size, debug=False):

    model.eval()
    batch_size = int(batch_size*1.5) #since you don't have to backward, you can have bigger batch

    assert dataset_mode in ['test', 'val']
    dataset = CleveRDataset(mode=dataset_mode)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8)

    count_success = 0

    if debug:
        sample_generator = enumerate(dataloader)
    else:
        sample_generator = tqdm.tqdm(enumerate(dataloader),total=len(dataloader))

    for num_batch, batch in sample_generator:

        if debug and num_batch > 50:
            break

        x, y = prepare_variables(batch, test_time=True)
        yhat = model.forward(x)
        count_success += count_good_prediction(yhat=yhat, y=y)

    surrog_len_dataset = (num_batch+1) * batch_size
    logger.info("Accuracy val (estim) : {:.2}".format(count_success/surrog_len_dataset))
    return count_success/len(dataset)


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


def prepare_variables(batch, test_time=False):

    x = dict()
    x['vision'] = Variable(batch['image'].type(FloatTensor), volatile=test_time)
    x['second_modality'] = Variable(batch['question'].type(LongTensor), volatile=test_time)

    y = Variable(batch['answer'][:,0].type(LongTensor), volatile=test_time)

    return x,y

def _prepare_variables(batch, y, total_number_task, num_task):
    """
    Prepare all torch.Tensor, cuda, and Variable
    """

    batch = FloatTensor(batch)
    y = Variable(LongTensor(y)).squeeze()
    task = torch.zeros(batch.size(0), total_number_task).type(FloatTensor)
    task[:, num_task] = 1

    x = {'vision': Variable(batch), 'second_modality': Variable(task)}

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
    parser.add_argument("-debug", type=bool, default=False, help="If true, less epochs and iterations")

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

    full_train_test(config=config, debug=args.debug)