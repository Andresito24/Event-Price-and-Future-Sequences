import torch
import torch.nn as nn
import torch.optim as optim
import math
print("Versión de PyTorch:", torch.__version__)
print("¿GPU disponible?", torch.cuda.is_available())

class MyRNN(nn.Module) :
    def __int__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.rnn = nn.RNN (
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )