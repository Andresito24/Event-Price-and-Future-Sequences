import torch
import torch.nn as nn
import torch.optim as nn
import math
print("Versión de PyTorch:", torch.__version__)
print("¿GPU disponible?", torch.cuda.is_available())

def CrearEventos() :
    elementos = []
    precios = []