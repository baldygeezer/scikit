
import sys
from time import time
from sklearn.metrics import accuracy_score
from sklearn import tree
import data


trainingTargets, trainingFeatures=data.processData('output_iow.csv')
testTargets, testFeatures = data.processData('results.csv')


dt = tree.DecisionTreeClassifier( )
dt.fit(trainingFeatures,trainingTargets)


dtPred = dt.predict(testFeatures)
acc = accuracy_score(dtPred, testTargets)
print acc

