import pandas as pd
import torch
import torch.nn as nn

from movieDataset import movieData
from torch.utils.data import DataLoader

class NCF(nn.Module):
	def __init__(self, num_users, num_movies,
		mf_dim, mlp_dim):
		super(NCF, self).__init__()
		self.user_mf = nn.Embedding(num_users+1, mf_dim)
		self.movie_mf = nn.Embedding(num_movies+1, mf_dim)

		self.user_mlp = nn.Embedding(num_users+1, mlp_dim)
		self.movie_mlp = nn.Embedding(num_movies+1, mlp_dim)

		self.layer1 = nn.Linear(2*mlp_dim, 2*mlp_dim)
		self.layer2 = nn.Linear(2*mlp_dim, 2*mlp_dim)
		self.layer3 = nn.Linear(2*mlp_dim, mlp_dim)
		self.final = nn.Linear(mlp_dim + mf_dim, 1)

	def forward(self, X):
		user = X[:,0]
		movie = X[:,1]
		umf = self.user_mf(user)
		mmf = self.movie_mf(movie)
		umlp = self.user_mlp(user)
		mmlp = self.movie_mlp(movie)

		mlp_input = torch.cat((umlp,mmlp), 1)

		out = nn.ReLU()(self.layer1(mlp_input))
		out = nn.ReLU()(self.layer2(out))
		out = nn.ReLU()(self.layer3(out))

		gmf = umf*mmf
		final_input = torch.cat((gmf, out), 1)

		final_out = self.final(final_input)
		final_out = 5*nn.Sigmoid()(final_out)
		return final_out

	def train(self, data, epochs, batch_size=30, lr=0.001):
		opt = torch.optim.Adam(self.parameters(), lr=lr)
		criterion = nn.MSELoss()
		train = movieData(data, train=True)
		test = movieData(data, train=False)

		trainLoader = DataLoader(dataset=train, batch_size=batch_size)
		testLoader = DataLoader(dataset=test, batch_size=batch_size)

		history = {'loss':[],'val_loss':[]}
		for epoch in range(epochs):

			avg_loss = []
			for X, y in trainLoader:
				opt.zero_grad()
				y_hat = self.forward(X)
				loss = criterion(y_hat, y)
				loss.backward()
				opt.step()
				history['loss'].append(loss.item())
				avg_loss.append(float(loss.item()))

			avg_val_loss = []
			for X, y in testLoader:
				y_hat = self.forward(X)
				val_loss =criterion(y_hat, y)
				history['val_loss'].append(val_loss.item())
				avg_val_loss.append(float(val_loss.item()))

			print(f"loss = {sum(avg_loss)/len(avg_loss)}", end=" ")
			print(f"val_loss = {sum(avg_val_loss)/len(avg_val_loss)}")
		return history
