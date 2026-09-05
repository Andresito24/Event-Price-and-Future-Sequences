import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
print("Versión de PyTorch:", torch.__version__)
print("¿GPU disponible?", torch.cuda.is_available())

class RNNDataset(Dataset) :
    def __init__(self):
        xy = np.loadtxt('sequences.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, :-1]).unsqueeze(-1) # all the samples
        self.y = torch.from_numpy(xy[:, -1]).unsqueeze(-1) # all the answers
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples
    
dataset = RNNDataset()
dataload = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

class MyRNN(nn.Module) :
    def __int__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.rnn = nn.RNN (
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) :
        output, hidden = self.rnn(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction