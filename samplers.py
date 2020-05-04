import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class LinearSampler():
    def __init__(self, batch_size, total_size) :

        self.batch_size = batch_size
        self.total_size = total_size

    def __len__(self):
        return int(self.total_size / self.batch_size)

    def __iter__(self):
        len = int(self.total_size / self.batch_size)
        perm = torch.randperm(self.total_size)

        for i in range(len) :
            start = i * self.batch_size
            end = start + self.batch_size
            yield perm[start : end]
