import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import math
data_panda=pd.read_csv('D:\TIPE\Python\Data TIPE\meteonet-master\meteonet-master\data_samples\ground_stations\SE_20160101.csv',sep=',')
data_array=data_panda.to_numpy()
data=np.zeros((111623,14),dtype=float)
num_utile=[]

for i in range(111623):
    if isinstance(data_array[i][-1],(int,float)) and isinstance(data_array[i][-1],(int,float)):
        num_utile.append(i)

for i in num_utile:
    date=data_array[i][4][:8]
    hour,minute=data_array[i][4][9:].split(':')
    data[i]=np.array(data_array[i][:4].tolist()+[date,hour,minute]+data_array[i][5:].tolist(),dtype=float)
data_tensors=torch.from_numpy(data)
train_dl = DataLoader(data_tensors,100, shuffle=True)
print(train_dl)


def lol():
    dtype = torch.float
    device = torch.device("cpu")
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None
    for chunk in chunk_container:
        a=1
