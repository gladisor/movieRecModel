from NCFModel import NCF

import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils import shuffle

BATCH_SIZE = 20
MF_DIM = 10
MLP_DIM = 5
EPOCHS = 10

data = pd.read_csv('datasets/ratings.csv')
data = data[['userId','movieId','rating']]
data = data.values
data = shuffle(data)

num_users = int(max(data[:,0]))
num_movies = int(max(data[:,1]))

model = NCF(num_users, num_movies, MF_DIM, MLP_DIM)
history = model.train(data=data, epochs=EPOCHS, 
	batch_size=BATCH_SIZE, lr=0.001)

import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(history['loss'])
ax1.set_title("Loss")
ax2.plot(history['val_loss'])
ax2.set_title("Val_loss")
plt.show()
