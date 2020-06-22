filename = 'BRCA_prognosis.txt'
with open(filename, 'r') as fin:
    lines = fin.readlines()

samples = lines[0].rstrip().split('\t')[1:]
labels = lines[1].rstrip().split('\t')[1:]
genes = []
data = []

for line in lines[2:]:
    tmp = line.rstrip().split('\t')
    genes.append(tmp[0])
    data.append(tmp[1:])

import numpy as np
data = np.array(data, dtype=np.float32).T
labels = np.array(labels, dtype=np.float32)

from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

from sklearn.cluster import KMeans
N = 2
kmeans = KMeans(n_clusters=N)
kmeans.fit(data_train)
centers_cluster = kmeans.cluster_centers_
cluster_train = kmeans.labels_
cluster_test = kmeans.predict(data_test)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(data_train)
scatter_train = pca.transform(data_train)
scatter_center = pca.transform(centers_cluster)

for i in range(N):
    X = scatter_train[cluster_train==i][:, 0]
    Y = scatter_train[cluster_train==i][:, 1]
    plt.scatter(X, Y, label='cluster[{}]'.format(i))
plt.scatter(scatter_center[:, 0], scatter_center[:, 1], c='k', marker='x', label='center')
plt.legend(loc='best')
plt.show()
