import pandas as pd
import torch
import torch.nn as nn

class NCF(nn.Module):
	def __init__(self, num_users, num_movies,
		mf_dim, mlp_dim):
		super(NCF, self).__init__()
		self.user_mf = nn.Embedding(num_users+1, mf_dim)
		self.movie_mf = nn.Embedding(num_movies+1, mf_dim)

		self.user_mlp = nn.Embedding(num_users+1, mlp_dim)
		self.movie_mlp = nn.Embedding(num_movies+1, mlp_dim)

		self.layer1 = nn.Linear(2*mlp_dim, 2*mlp_dim)
		self.layer2 = nn.Linear(2*mlp_dim, mlp_dim)
		self.final = nn.Linear(mlp_dim + mf_dim, 1)

	def forward(self, X):
		user = X[:,0]
		movie = X[:,1]
		umf = self.user_mf(user)
		mmf = self.movie_mf(movie)
		umlp = self.user_mlp(user)
		mmlp = self.movie_mlp(movie)

		mlp_input = torch.cat((umlp,mmlp), 1)

		out = self.layer1(mlp_input)
		out = self.layer2(out)

		gmf = umf*mmf
		final_input = torch.cat((gmf, out), 1)

		final_out = self.final(final_input)
		final_out = nn.Sigmoid()(final_out)
		return final_out

if __name__ == "__main__":
	data = pd.read_csv('datasets/ratings.csv')
	data = data[['userId','movieId','rating']]
	data = data.values

	X = data[:,[0,1]]
	y = data[:,2]

	X = torch.tensor(X).long()
	y = torch.tensor(y)

	num_users = torch.max(X[:,0]).item()
	num_movies = torch.max(X[:,1]).item()

	mf_dim = 10
	mlp_dim = 5

	model = NCF(num_users, num_movies, mf_dim, mlp_dim)

	print(model(X))
