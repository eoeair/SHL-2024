import torch
import pickle
import numpy as np

class Feeder_modal(torch.utils.data.Dataset):
    """ Feeder for modal inputs """

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        data = np.load(self.data_path,mmap_mode='r')
        
        self.data = data.transpose(0,2,1,3).reshape(-1,data.shape[1],data.shape[3])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # get data
        lenth = int(self.data.shape[0] / 4)
        if index in range(0, lenth * 1):
            return self.data[index], 0
        elif index in range(lenth, lenth * 2):
            return self.data[index], 1
        elif index in range(lenth * 2, lenth * 3):
            return self.data[index], 2
        else:
            return self.data[index], 3

class Feeder_label(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, label_path, modal):
        self.data_path = data_path
        self.label_path = label_path
        self.modal = modal
        self.load_data()
    
    def load_data(self):
        self.data = np.load(self.data_path)[self.modal].transpose(1,0,2)
        self.label = np.load(self.label_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data
        return self.data[index], self.label[index]