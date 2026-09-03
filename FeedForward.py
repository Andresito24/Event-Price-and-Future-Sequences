import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
print("Versión de PyTorch:", torch.__version__)
print("¿GPU disponible?", torch.cuda.is_available()) 

def crearCasas(cantidad):
    datos = []
    precios = []
    
    for i in range(cantidad) :

        metrosCasa = random.uniform(50,250)
        habitaciones = random.randint(1,6)
        edad = random.uniform(0,100)
        ubicacion = random.randint(1,5)
        baños = random.randint(1,5)
        metrosTerreno = random.uniform(metrosCasa, metrosCasa*random.uniform(1,3))
        calidadConstruccion = random.randint(1,10)
        mantenimiento = random.randint(1,10)
        centroDistancia = random.uniform(1,50)
        precioPromedio = random.uniform(500000,15000000)
        calidadAire = random.randint(1,10)
        diseñoModerno = random.randint(1,10)
        alturaTecho = random.uniform(2.2,5)
        extras = random.randint(0,5)

        precio = (
            metrosCasa * 12000 +
            ubicacion * 150000 +
            habitaciones * 180000 +
            baños * 250000 +
            metrosTerreno * 5000 -
            edad * 15000 +
            calidadConstruccion * 200000 +
            mantenimiento * 100000 -
            centroDistancia * 50000 +
            extras * 300000 +
            precioPromedio * 0.10 +
            calidadAire * 80000 + 
            diseñoModerno * 90000 +
            alturaTecho * 80000
        )

        ubicacion = (ubicacion - 1) / (5 - 1)
        habitaciones = (habitaciones - 1) / (6 - 1)
        baños = (baños - 1) / (5 - 1)
        metrosTerreno = (metrosTerreno - 50) / (750 - 50)
        metrosCasa = (metrosCasa - 50) / (250 - 50)
        edad = edad / 100
        calidadConstruccion = (calidadConstruccion - 1) / 9
        mantenimiento = (mantenimiento - 1) / 9
        centroDistancia = (centroDistancia - 1) / 49
        extras = (extras - 0) / 5
        precioPromedio = (precioPromedio - 500000) / (15000000 - 500000)
        calidadAire = (calidadAire - 1) / 9
        diseñoModerno = (diseñoModerno - 1) / 9
        alturaTecho = (alturaTecho - 2.2) / (5 - 2.2)

        datos.append([ metrosCasa, habitaciones, edad, ubicacion, baños, metrosTerreno, calidadConstruccion, mantenimiento, centroDistancia, extras, precioPromedio, calidadAire, diseñoModerno, alturaTecho])
        precios.append(precio)

        

    return datos, precios

X, y = crearCasas(5000)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
y = y.reshape(-1,1)

precioMin = y.min()
precioMax = y.max()
y = (
    (y-precioMin) / (precioMax-precioMin)
)

class PrecioCasa(nn.Module) :
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(14,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,1),
        )

    def forward(self,x):
        return self.network(x)

modelo = PrecioCasa()
criterion = nn.MSELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)
epochs = 500

for epoch in range(epochs) :
    optimizer.zero_grad()
    predict = modelo(X)
    loss = criterion(predict, y)
    loss.backward()
    optimizer.step()


Xtest, ytest = crearCasas(500)    
Xtest = torch.tensor(Xtest, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.float32)
ytest = ytest.reshape(-1,1)
precioMintest = precioMin
precioMaxtest = precioMax
ytest = (
    (ytest-precioMintest) / (precioMaxtest-precioMintest)
)
modelo.eval()

with torch.no_grad(): 
    predictTest = modelo(Xtest)

predictReal = predictTest * (precioMax - precioMin) + precioMin
ytestReal = ytest * (precioMax - precioMin) + precioMin

mae = torch.mean(torch.abs(predictReal - ytestReal))
print("Error promedio:", mae.item())

error_porcentaje = (
    mae / ytestReal.mean()
) * 100

print("Error porcentual:", error_porcentaje.item(), "%")

Xprueba = torch.tensor([
   0, # metrosCasa
   0, # habitaciones
   1, # edad
   0, # ubicacion
   0, # baños
   0, # metrosTerreno
   0, # calidadConstruccion
   0, # mantenimiento
   1, # centroDistancia
   0, # extras
   0, # precioPromedio
   0, # calidadAire
   0, # diseñoModerno
   0 # alturaTecho
], dtype=torch.float32)
predictPrueba = modelo(Xprueba)
predictPrueba = predictPrueba * (precioMax - precioMin) + precioMin
print(predictPrueba)