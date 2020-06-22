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

from sklearn.decomposition import PCA

model = PCA(n_components=2)
model.fit(data_train)

data_train = model.transform(data_train)
data_test = model.transform(data_test)

import matplotlib.pyplot as plt

X = data_train[:, 0]
Y = data_train[:, 1]
plt.scatter(X, Y, marker='o', label='Train')

X = data_test[:, 0]
Y = data_test[:, 1]
plt.scatter(X, Y, marker='x', label='Test')

plt.legend(loc='best')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.show()