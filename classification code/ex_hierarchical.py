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

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=5)
clf = clf.fit(data_train, labels_train)

model = SelectFromModel(clf, threshold=0.01, prefit=True)
data_train = model.transform(data_train)
features = [genes[i] for i in model.get_support(indices=True)]

from scipy.cluster.hierarchy import linkage
Z = linkage(data_train.T, method='single', metric='correlation')

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

plt.xlabel('Gene Symbol')
plt.ylabel('Distance')
dendrogram(Z, labels=features, color_threshold=0.6)
plt.axhline(y=0.6, c='k', linestyle='--')
plt.show()