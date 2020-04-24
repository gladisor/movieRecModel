import torch
import torch.nn as nn

## mf stands for Matrix Factorization
# We embed users and movies into an n dimentional latent space
# then take the dot product and concatenate the resulting
# vector with the output of the mlp.

## mlp stands for Multi Layer Perceptron
# First we embed the user and movie into an m dimentional latent
# space. Then we concatenate the two m dimentional vectors
# and pass it through a neural net.

## Finally we concatenate the n dimentional mf vector
# and the m dimentional mlp vector and pass it through a final layer.

class NCF(nn.Module):
	def __init__(self, num_users, num_movies, mf_dim, mlp_dim):
		super(NCF, self).__init__()
		self.user_mf = nn.Embedding(num_users, mf_dim)
		self.movie_mf = nn.Embedding(num_movies, mf_dim)

		self.user_mlp = nn.Embedding(num_users, mlp_dim)
		self.movie_mlp = nn.Embedding(num_movies, mlp_dim)

		self.mlp1 = nn.Linear(mlp_dim, mlp_dim)

		self.final = nn.Linear(mlp_dim, 1)

	def forward(self, x):
		u_mf = self.user_mf(x[:,0])
		m_mf = self.movie_mf(x[:,1])
		gmf = u_mf * m_mf

		u_mlp = self.user_mlp(x[:,0])
		m_mlp = self.movie_mlp(x[:,1])

		mlp = torch.cat((u_mlp, m_mlp), dim=1)
		out = self.mlp1(mlp)
		out = nn.ReLU()(out)

		final = torch.cat((gmf, out), dim=1)

		out = self.final(final)
		out = nn.Sigmoid()(out)
		return out

import pandas as pd

data = pd.read_csv('reduced.csv')

num_users = data['userId'].max()
num_movies = data['movieId'].max()

hist = []
for user in data['userId'].unique():
	hist.append(len(data.loc[data['userId'] == user]))

import matplotlib.pyplot as plt

plt.hist(hist)
plt.show()
