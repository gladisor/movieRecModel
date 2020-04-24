import pandas as pd

data = pd.read_csv('datasets/reduced.csv')

num_users = data['userId'].max()
num_movies = data['movieId'].max()

users = []
for user in data['userId'].unique():
	users.append(len(data.loc[data['userId'] == user]))

movies = []
for movie in data['movieId'].unique():
	movies.append(len(data.loc[data['movieId'] == movie]))

import matplotlib.pyplot as plt

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

# Most users have reviewed 0 to 100 movies
ax1.hist(users)
ax1.set_title("Number of users who reviewed x movies")
ax1.set_xlabel("Number of movies a user has reviewed")

ax2.hist(movies)
ax2.set_title("Number of movies that have x reviews")
ax2.set_xlabel("Number of reviews an movie has")
plt.show()