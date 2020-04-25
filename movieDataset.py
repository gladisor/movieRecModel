from torch.utils.data import Dataset
import torch

class movieData(Dataset):
	def __init__(self, data, train):
		X = data[:,[0,1]]
		y = data[:,2]

		X = torch.tensor(X).long()
		y = torch.tensor(y).float().view(y.shape[0],1)

		size = X.shape[0]
		partition = int(size*0.9)

		if train:
			self.X = X[0:partition]
			self.y = y[0:partition]
		else:
			self.X = X[partition:X.shape[0]]
			self.y = y[partition:y.shape[0]]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

	def __len__(self):
		return len(self.y)