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
features = [genes[i] for i in model.get_support(indices=True)]
scores = [clf.feature_importances_[i] for i in model.get_support(indices=True)]

from operator import itemgetter
print('n_features: {}'.format(len(features)))
for feature, score in sorted(list(zip(features, scores)), key=itemgetter(1), reverse=True):
    #print('{}\t{:.4f}'.format(feature, score))
    print(feature)

print('[before] data_train: {} x {}'.format(*data_train.shape))
data_train = model.transform(data_train)
print('[after] data_train: {} x {}'.format(*data_train.shape))