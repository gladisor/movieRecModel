from NCFModel import NCF
import pandas as pd
import torch
import torch.nn as nn

data = pd.read_csv('datasets/ratings.csv')
data = data[['userId','movieId','rating']]
data = data.values

X = data[:,[0,1]]
y = data[:,2]

X = torch.tensor(X).long()
y = torch.tensor(y).float()

num_users = torch.max(X[:,0]).item()
num_movies = torch.max(X[:,1]).item()

mf_dim = 10
mlp_dim = 5

model = NCF(num_users, num_movies, mf_dim, mlp_dim)
opt = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

epochs = 100

history = []
for epoch in range(epochs):
	opt.zero_grad()
	y_hat = model(X).view(-1)
	loss = criterion(y_hat, y)
	loss.backward()
	opt.step()
	print(loss.item())
	history.append(loss.item())

import matplotlib.pyplot as plt
plt.plot(history)
plt.show()
