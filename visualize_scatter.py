import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

train_labels=np.load('labels.npy')
train_features=np.load('features.npy')

X=train_features
X=X/max(X.ravel())
y=train_labels
for k in range(0,10):
	print(len(np.where(y==k)[0]))
for i, j in enumerate(np.unique(y)):
   	plt.scatter(X[y == j, 0], X[y == j, 1],c = ListedColormap(('black', 'red', 'gray', 'yellow', 'silver', 'blue','magenta', 'brown', 'purple', 'orange'))(i), label = j)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.savefig('scatter_plot.png')