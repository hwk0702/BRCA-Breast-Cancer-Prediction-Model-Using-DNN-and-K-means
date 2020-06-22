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

from sklearn.neural_network import MLPClassifier
'''
clf = MLPClassifier(hidden_layer_sizes=[1024, 256, 32],
                    activation='logistic',
                    solver='adam',
                    learning_rate_init=0.01)
'''




clf.fit(data_train, labels_train)

prediction_test = clf.predict(data_test)
probability_test = clf.predict_proba(data_test)[:,1]

from sklearn.metrics import confusion_matrix, roc_auc_score

auc = roc_auc_score(labels_test, probability_test)

tn, fp, fn, tp = confusion_matrix(labels_test, prediction_test).ravel()
accuracy = (tp + tn) / (tn + tp + fn + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (fp + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = 2 * precision * recall / (precision + recall)

print(tn, fp, fn, tp)
print(accuracy)
print(sensitivity)
print(specificity)
print(precision)
print(recall)
print(fscore)

print(labels_test)