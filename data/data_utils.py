import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class GCDClassUniformlySampler(Sampler):
    '''
    random sample according to class label
    Arguments:
        train_index_mapper: a tensor that maps each unique image index, to its position in the whole train set
        img_cluster_index: a tensor that records the cluster label for each unique image index
    '''
    def __init__(self, train_index_mapper, cluster_img_dict, k=16):

        for lbl in cluster_img_dict.keys():
            img_idxs = cluster_img_dict[lbl]
            cluster_img_dict[lbl] = list(train_index_mapper[img_idxs].numpy())

        self.cluster_img_dict = cluster_img_dict
        self.k = k

    def __iter__(self):
        self.sample_list = self._generate_list(self.cluster_img_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _generate_list(self, id_dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''
        sample_list = []
        dict_copy = id_dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)

        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k    # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset, return_path=False):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None
        self.return_path = return_path

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            if self.return_path:
                img, label, uq_idx, path = self.labelled_dataset[item]
            else:
                img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            if self.return_path:
                img, label, uq_idx, path = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            else:
                img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        if self.return_path:
            return img, label, uq_idx, np.array([labeled_or_not]), path
        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
