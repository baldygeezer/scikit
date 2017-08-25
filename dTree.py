
from sklearn.metrics import accuracy_score
from sklearn import tree
import data


v=19


print "\n"
def runMLFullMFD(v):
    trainingTargets, trainingFeatures = data.processData('winSet.csv')
    testTargets, testFeatures = data.processData('iowAset.csv')
    dt = tree.DecisionTreeClassifier()
    dt.fit(trainingFeatures, trainingTargets)
    dtPred = dt.predict(testFeatures)
    acc = accuracy_score(dtPred, testTargets)

    print "*****************************************************"
    print "Accuracy using MFD, #vertices, #edges and diameter"
    print " params: max_depth=" + str(v)
    print acc
    print "*****************************************************"

def runNumVE_D(v):
    trainingTargets2, trainingFeatures2 = data.processDataNoMFD('winSet.csv')
    testTargets2, testFeatures2 = data.processDataNoMFD('iowAset.csv')
    dt2 = tree.DecisionTreeClassifier(max_leaf_nodes=2)
    dt2.fit(trainingFeatures2, trainingTargets2)
    dtPred2 = dt2.predict(testFeatures2)
    acc2 = accuracy_score(dtPred2, testTargets2)

    print "*****************************************************"
    print "Accuracy using just #vertices, #edges and diameter"
    print " params: max_depth=" + str(v)
    print acc2
    print "*****************************************************"

for v in range(0,25,1):
    runMLFullMFD(v)




#tree.export_graphviz(dt,out_file=data.datapath+"treeMFD.dot")
#tree.export_graphviz(dt,out_file=data.datapath+"treeNoMFD.dot")