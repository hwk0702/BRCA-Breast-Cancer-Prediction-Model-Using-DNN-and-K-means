# BRCA Breast Cancer Prediction Model Using DNN and K-means

### 1. Introduction

#### 1.1 Set goals for project

- Use BRCA_prognosis data to create a program that can detect gene anomalies and predict breast cancer.

①	Execute test data and training data separately.

②	Using gene data from patients, distinguishes between good and risky genes. 

③	As a result, the program will be makes good and risky predictions of the gene.

#### 1.2 Data description

- Data structure

<img src='/img/BRCA_data1.png' width='500'>

<img src='/img/BRCA_data2.png' width='500'>

---

### 2. Model training with sklearn

- Without preprocessing, the results of unsupervised learning

i.	KNN (k=[3, 5, 7, 9, 11, 13, 15])

ii.	Naive Bayesian Classification

iii.	Information gain ( max_depth=[3, 5, 7, 9, 11, 13, 15] )

iv.	SVM (kernel = [linear,  poly,  rbf,  sigmoid])

v.	DNN (solver=[adam, sgd, lbfgs], activation= [identity,  logistic,  tanh,  relu]) 

<img src='/img/BRCA1.png' width='400'>

The accuracy of DNN was the highest at about 0.85, and DNN had the highest value except the sensitivity. As a result, DNN (solver = ibfgs, activation = logistic) is the best classification.

When SVM kernel is sigmoid and DNN solver is adman and sgd, it is not classified properly.

---

### 3. Method

1)	Preprocessing

①	Edit labels array

For use in the DNN model, the Labels array is changed to a two-dimensional array, and the column vectors are replaced by row vectors.

<img src='/img/BRCA2.PNG' width='400'>

②	One-Hot-Encoding

One-Hot-Encoding is used to change the values of labels. One-Hot-Encoder is also referred to as One-of-K encoding and converts an integer scalar value having a value of 0 to K-1 into a K-dimensional vector having a value of 0 or 1.

<img src='/img/BRCA3.png' width='400'>

③	Normalization

Normalize the values of the data. Normalization is a transformation to make all of the individual data the same size.

<img src='/img/BRCA4.png' width='400'>


2)	Data grouping

Divide into two groups with similar characteristics to get better results.

①	 Principal component analysis (PCA)

The dimension of the data is reduced to two dimensions.

<img src='/img/BRCA5.png' width='400'>

①	 K-means

Use K-means to divide into two groups.

<img src='/img/BRCA6.png' width='400'>

③	Grouping

<img src='/img/BRCA7.png' width='400'>

3)	Training

①	DNN (Deep Neural Network)

<img src='/img/BRCA8.png' width='400'>

The hidden layer is composed of four layers (4096, 1024, 256, 32) and the Learning_rate is set to 0.0001. I used the solver as the adam optimizer function and the activate function as relu. Train step were set to 500.

②	Dropout

<img src='/img/BRCA9.png' width='400'>

Avoid using some of the neurons at each learning step to prevent some features from sticking to specific neurons, balancing the weights to prevent overfitting.

Dropouts were set to 0.8.

③	Regularization

<img src='/img/BRCA10.png' width='400'>

Let’s not have too big numbers in the weight. And, prevent overfitting. 

Reularization was set to 0.001.

---

### 4. result

1)	Result

①	First Group & Second Group 

<img src='/img/BRCA11.png' width='400'>

②	Sum first group, seconde group result

<img src='/img/BRCA12.png' width='400'>

<img src='/img/BRCA13.png' width='400'>

<img src='/img/BRCA14.png' width='100'>

Accuracy was 0.88

2)	Compare with other methods

①	Not grouping, Not regularization, node(1024,256,32)
 
<img src='/img/BRCA15.png' width='400'>

②	Not grouping, Not regularization, node(4096,1024,256,32)
 
<img src='/img/BRCA16.png' width='400'>

③	Not grouping, Use regularization, node(4096,1024,256,32)
 
<img src='/img/BRCA17.png' width='400'>

