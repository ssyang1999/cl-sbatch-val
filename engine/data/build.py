from .common import ConcatDataset
import engine.data.datasets as D

import os


class DatasetCatalog(object):
    DATA_DIR = "../datasets"
    DATASETS = {
        "cifar10": {
            "root": "cirar/",
            "download": True,
        }
    }

    @staticmethod
    def get(name):
        if "cifar" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                download=attrs["download"]
            )
            return dict(
                factory="CIFAR10",
                args=args,
            )
        # elif "your dataset name" in name:
        # data_dir = DatasetCatalog.DATA_DIR
        # attrs = DatasetCatalog.DATASETS[name]
        # args = dict(
        #     data_dir=os.path.join(data_dir, attrs["data_dir"]),
        #     split=attrs["split"],
        # )
        # return dict(
        #     factory="your data set implementation",
        #     args=args,
        # )
        raise RuntimeError("Dataset not available: {}".format(name))


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
        Arguments:
            dataset_list (list[str]): Contains the names of the datasets, i.e.,
                coco_2014_train, coco_2014_val, etc
            transforms (callable): transforms to apply to each (image, target) sample
            dataset_catalog (DatasetCatalog): contains the information on how to
                construct a dataset.
            is_train (bool): whether to setup the dataset for training or testing
        """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        # if "CIFAR" in data["factory"]:
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset
