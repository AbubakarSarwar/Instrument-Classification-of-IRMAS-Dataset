import pandas as pd
import itertools

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import numpy as np
from sklearn.model_selection import LeaveOneOut

# pca use kya bt features reduce nhi hue
#chi2, variance threshold mein input -ve nhi hona chahye 
#L1 based features selection apply kya hai


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def Comparing_accuracy(name, model):
    ########################################## K FOLD ################################
        
    print(name)
    kf = cross_validation.KFold(n = 1759, n_folds=10)
    
    accuracy = []
    class_names={'Acoustic Guitar', 'Trumpet','Volume','Electric Guitar'}
    
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        
        X_train, X_test = training[train_index], training[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model.fit(X_train,y_train)
        prediction = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))
    
    Mean_Accuracy = np.mean(accuracy)
    print("K-Fold:  Mean Accuracy: " + str(round(Mean_Accuracy,3)))
    cnf_matrix = confusion_matrix(y_test, prediction)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

    plt.figure()    
        
    ########################################### LEAVE OUT ONE #################################
    """
    accuracy = []

    loo = LeaveOneOut()
    loo.get_n_splits(training)
    
    for train_index, test_index in loo.split(training):
        #print("TRAIN:", train_index, "TEST:", test_index)
        
        X_train, X_test = training[train_index], training[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model.fit(X_train,y_train)
        prediction = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))
        
    Mean_Accuracy = np.mean(accuracy)
    print("Leave Out One:  Mean Accuracy: " + str(round(Mean_Accuracy,3)))
    """
    
    
    
    ############################################################ HOLDOUT ##############################
    
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(training, labels, test_size = 0.3, random_state = 7)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print("Hold Out: Mean Accuracy: "+ str(round(accuracy_score(Y_test, predictions),3)))

    
    return



dataset = pd.read_csv("Instrument.csv")

array = dataset.values
training = array[:,4:102]
labels = dataset ['instrument_label']


print(training.shape)


lsvc = LinearSVC(C=0.000001, penalty="l1", dual=False).fit(training, labels)
model = SelectFromModel(lsvc, prefit=True)
train_new = model.transform(training)
print(train_new.shape)


Bagging = BaggingClassifier()

Classifier = [Bagging]
    
for j in Classifier:
    if j == Bagging:
        name = "Bagging"

    Comparing_accuracy(name, j)

