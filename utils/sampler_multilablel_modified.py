import random
import numpy as np
import sys
from torch.utils.data.sampler import Sampler

#Inspired from : https://github.com/issamemari/pytorch-multilabel-balanced-sampler/blob/b11b7fa0d0334795bdcc600812d1edbe2ce7ea7b/sampler.py#L7

class MultilabelBalancedRandomSampler(Sampler):
    def __init__(self, labels, indices=None):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)

            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
        """
        self.labels = labels
        self.indices = indices
        self.current_class = 0

        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1] #number of cell lines
        # List of lists of example indices per class
        self.class_indices = []
        self.not_class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

            lst_not = np.where(self.labels[:, class_] == 0)[0]
            lst_not = lst_not[np.isin(lst_not, self.indices)]
            self.not_class_indices.append(lst_not)


    def __iter__(self):
        """
            Provides a way to iterate over indices of dataset elements
        """
        self.count = 0
        return self

    #next() called on an iterator object to yield the next item in the sequence.
    # len(self.indices) is the length of the training set
    # we will loop through all items of the training set
    # where for each item we will have 2 instances corresponding to a specific cell line
    # we stop the iteration through our dataloader when the number of times sample is called = length of training set
    def __next__(self):
        #print("Next " + str(self.count) + " " + str(len(self.indices)))
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        #for each cell line sequentially select an instance randomly that gives positive and another negative
        class_ = self.current_class
        self.current_class = (self.current_class + 1) % self.labels.shape[1]

        class_indices = self.class_indices[class_]
        not_class_indices = self.not_class_indices[class_]

        chosen_index_pos = np.random.choice(class_indices)
        chosen_index_neg = np.random.choice(not_class_indices)
        lst_idx = [chosen_index_neg, chosen_index_pos]
        lst_idx.sort()
        #print(lst_idx, class_)
        return lst_idx, class_

    def __len__(self):
        """
            Returns the length of the returned iterators
            *change
        """
        return len(self.indices)
