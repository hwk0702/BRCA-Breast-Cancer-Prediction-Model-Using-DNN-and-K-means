######## 1. Read data
filename = 'BRCA_prognosis.txt'
with open(filename, 'r') as fin:
    lines = fin.readlines()
## 1-1) Parsing
samples = lines[0].rstrip().split('\t')[1:]
labels = lines[1].rstrip().split('\t')[1:]
genes = []
data = []
for line in lines[2:]:
    tmp = line.rstrip().split('\t')
    genes.append(tmp[0])
    data.append(tmp[1:])
## 1-2) Converting to numpy format
import numpy as np
data = np.array(data, dtype=np.float32).T # transpose: by gene --> by sample
labels = np.array(labels, dtype=np.float32)
######## 2. Split into training and test datasets
from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
test_size=0.2, stratify=labels)
######## 3. Make a classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
## 3-1) training
clf.fit(data_train, labels_train)
## 3-2) prediction
prediction_test = clf.predict(data_test)
probability_test = clf.predict_proba(data_test)[:,1]
## 4. Evaluate the model
from sklearn.metrics import confusion_matrix, roc_auc_score
## 4-1) AUC
auc = roc_auc_score(labels_test, probability_test)
## 4-2) Confusion matrix
tn, fp, fn, tp = confusion_matrix(labels_test, prediction_test).ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
sensitivity = tp / (tp + fn)
specificity = tn / (fp + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = 2 * precision * recall / (precision + recall)

print(specificity)