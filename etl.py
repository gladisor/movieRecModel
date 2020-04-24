import pandas as pd

data = pd.read_csv('ratings.csv')

data = data[['userId','movieId','rating']]

data = data[0:5000]

data.to_csv('reduced.csv', index=False)