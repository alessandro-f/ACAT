"""
Storage associated utilities
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.misc
import shutil
import torch
from collections import OrderedDict
import scipy
import json
import os


def isfloat(x):
    return isinstance(x, float)


def isint(x):
    return isinstance(x, int)


def save_metrics_dict_in_json(path, metrics_dict, overwrite):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath, if False adds metrics to existing
    """

    if path.endswith(".json"):
        path.replace(".json", "")

    metrics_file_path = path

    if overwrite and os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)

    with open("{}.json".format(metrics_file_path), "w+") as json_file:
        json.dump(metrics_dict, json_file)


def load_metrics_dict_from_json(path):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if not path.endswith(".json"):
        path = "{}.json".format(path)
    with open(path) as json_file:
        metrics_dict = json.load(json_file)

    return metrics_dict


def save_to_json(filename, dict_to_store):
    with open(os.path.abspath(filename), "w") as f:
        json.dump(dict_to_store, fp=f, indent=4)


def load_dict_from_json(path):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if not path.endswith(".json"):
        path = "{}.json".format(path)
    with open(path) as json_file:
        metrics_dict = json.load(json_file)

    return metrics_dict


def load_from_json(filename):
    with open(filename, mode="r") as f:
        load_dict = json.load(fp=f)

    return load_dict


def load_metrics_dict_from_pt(path):
    # sourcery skip: inline-immediately-returned-variable
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if not path.endswith(".pt"):
        path = "{}.pt".format(path)

    metrics_file_path = path

    metrics_dict = torch.load(metrics_file_path)

    return metrics_dict


def save_metrics_dict_in_pt(path, metrics_dict, overwrite):
    # sourcery skip: merge-nested-ifs
    """
    Saves a metrics .pt file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath, if False adds metrics to existing
    """
    if not path.endswith(".pt"):
        path = "{}.pt".format(path)

    metrics_file_path = path

    if overwrite:
        if os.path.exists(metrics_file_path):
            os.remove(metrics_file_path)

    torch.save(metrics_dict, metrics_file_path)


def save_checkpoint(state, is_best, directory="", filename="checkpoint.pth.tar"):
    """
    Checkpoint saving utility, to ensure that the checkpoints are saved in the right place
    :param state: this is what gets saved.
    :param is_best: if this is the current best model, save a copy of it with a `best_`
    :param directory: where to save
    :param filename: using this filename
    :return: nothing, just save things
    """


    if is_best:
        best_save_path = (
            "{}/best_{}".format(directory, filename)
            if directory != ""
            else "best_{}".format(filename)
        )
        torch.save(state, best_save_path)
        return best_save_path
        #shutil.copyfile(save_path, best_save_path)
    else:
        save_path = "{}/{}".format(directory, filename) if directory != "" else filename
        torch.save(state, save_path)
        return save_path


def restore_model_from_path(restore_fields, path, device="cpu"):
    if not os.path.isfile(path):
        return -1
    checkpoint = torch.load(
        path,
        map_location=lambda storage, loc: storage,
    )
    print("device: {} torch.cuda.device_count(): {}".format(device, torch.cuda.device_count()))
    for name, field in restore_fields.items():
        new_state_dict = OrderedDict()
        for k, v in checkpoint[name].items():
            if "module" in k and (
                device == "cpu" or torch.cuda.device_count() == 1
            ):
                name = k.replace("module.", "")  # remove module.
            else:
                name = k
            new_state_dict[name] = v

        field.load_state_dict(new_state_dict)
    return field.state_dict()


def restore_model(restore_fields, path, epoch=None, device="cpu", best = False):
    """
    Model restoration. This is built into the experiment framework and args.latest_loadpath should contain the path
    to the latest restoration point. This is automatically set in the framework
    :param net: Network to restore weights of
    :param optimizer: sometimes the optimizer also needs to be restored.
    :param args:
    :return: Nothing, only restore the network and optimizer.
    """
    if epoch is None:
       if best == False:
           checkpoint_name = "latest_ckpt.pth.tar"
       elif best == True:
           checkpoint_name = "best_ckpt.pth.tar"
    else:
        checkpoint_name = "{}_ckpt.pth.tar".format(epoch)

    # print(path, checkpoint_name)

    if os.path.isfile("{}/{}".format(path, checkpoint_name)):
        checkpoint = torch.load(
            "{}/{}".format(path, checkpoint_name),
            map_location=lambda storage, loc: storage,
        )
        print("device: {} torch.cuda.device_count(): {}".format(device, torch.cuda.device_count()))
        for name, field in restore_fields.items():
            new_state_dict = OrderedDict()
            for k, v in checkpoint[name].items():
                if "module" in k and (
                    device == "cpu" or torch.cuda.device_count() == 1
                ):
                    name = k.replace("module.", "")  # remove module.
                else:
                    name = k
                new_state_dict[name] = v


            field.load_state_dict(new_state_dict, strict = False)
            # print(field, 'restored')

            print('resumed from epoch:', checkpoint["epoch"] )

        return checkpoint["epoch"]
    else:
        print('failed to resume:')
        return -1


def build_experiment_folder(experiment_name, log_path, sub_folder=None, save_images=True):
    """
    An experiment logging folder goes along with each experiment. This builds that folder
    :param args: dictionary of arguments
    :return: filepaths for saved models, logs, and images
    """
    if sub_folder is not None:
        saved_models_filepath = os.path.join(
            log_path, experiment_name.replace("%.%", "/"), "saved_models", sub_folder
        )
        logs_filepath = os.path.join(
            log_path, experiment_name.replace("%.%", "/"), "summary_logs", sub_folder
        )
        images_filepath = os.path.join(
            log_path, experiment_name.replace("%.%", "/"), "images", sub_folder
        )
    else:
        saved_models_filepath = os.path.join(
            log_path, experiment_name.replace("%.%", "/"), "saved_models"
        )
        logs_filepath = os.path.join(
            log_path, experiment_name.replace("%.%", "/"), "summary_logs"
        )
        images_filepath = os.path.join(
            log_path, experiment_name.replace("%.%", "/"), "images"
        )

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    if save_images:
        if not os.path.exists(images_filepath + "/train"):
            os.makedirs(images_filepath + "/train")
        if not os.path.exists(images_filepath + "/val"):
            os.makedirs(images_filepath + "/val")
        if not os.path.exists(images_filepath + "/test"):
            os.makedirs(images_filepath + "/test")

    return saved_models_filepath, logs_filepath, images_filepath


def get_best_performing_epoch_on_target_metric(
    metrics_dict, target_metric, ranking_method=np.argmax
):
    """
    utility for finding best epoch thus far
    :param: metrics_dict: A dictionary containing the collected metrics from which to extract the best perf. model epoch
    target_metric:
    ranking_method:
    :return: best epoch, and what the best target metric value was
    """
    best_model_epoch = 0
    best_target_metric = None

    if target_metric in metrics_dict and len(metrics_dict[target_metric]) != 0:
        best_epoch_idx = ranking_method(metrics_dict[target_metric])
        best_model_epoch, best_target_metric = (
            metrics_dict["epoch"][best_epoch_idx],
            metrics_dict[target_metric][best_epoch_idx],
        )

    return best_model_epoch, best_target_metric


def print_network_stats(net):
    """
    Utility for printing how many parameters and weights in the network
    :param net: network to observe
    :return: nothing, just print
    """
    trainable_params_count = 0
    trainable_weights_count = 0
    for param in net.parameters():
        weight_count = 1
        for w in param.shape:
            weight_count *= w
        if param.requires_grad:
            trainable_params_count += 1
            trainable_weights_count += weight_count

    print(
        "{} parameters and {} weights are trainable".format(
            trainable_params_count, trainable_weights_count
        )
    )
