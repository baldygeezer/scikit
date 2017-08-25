from sklearn.metrics import accuracy_score
from sklearn import tree
import data
import csv






def runMLFullMFD(v, param):
    usingDefault = True
    if param=="max_depth":
       dt = tree.DecisionTreeClassifier(max_depth=v)
       usingDefault=False
    elif param=="max_leaf_nodes":
       dt = tree.DecisionTreeClassifier(max_leaf_nodes=v)
       usingDefault=False
    elif param=="mins_samples_split":
        dt = tree.DecisionTreeClassifier(min_samples_split=v)
    else:
       dt = tree.DecisionTreeClassifier()


    trainingTargets, trainingFeatures = data.processData('winSet.csv')
    testTargets, testFeatures = data.processData('iowAset.csv')

    dt.fit(trainingFeatures, trainingTargets)
    dtPred = dt.predict(testFeatures)
    acc = accuracy_score(dtPred, testTargets)
    label=param+"="+str(v)


    print "*****************************************************"
    print "Accuracy using MFD, #vertices, #edges and diameter"
    if usingDefault:
        print "parameter not recognised, using default values"
    else:
        print "using: " + param +" "+ str(v)
    print acc
    print "*****************************************************"
    return (acc,label)


def runNumVE_D(v,param):
    usingDefault = True
    if param == "max_depth":
        dt2 = tree.DecisionTreeClassifier(max_depth=v)
        usingDefault = False
    elif param == "max_leaf_nodes":
        dt2 = tree.DecisionTreeClassifier(max_leaf_nodes=v)
        usingDefault = False
    else:
        dt2 = tree.DecisionTreeClassifier()

    trainingTargets2, trainingFeatures2 = data.processDataNoMFD('winSet.csv')
    testTargets2, testFeatures2 = data.processDataNoMFD('iowAset.csv')

    dt2.fit(trainingFeatures2, trainingTargets2)
    dtPred2 = dt2.predict(testFeatures2)
    acc2 = accuracy_score(dtPred2, testTargets2)
    label = param + "=" + str(v)

    print "*****************************************************"
    print "Accuracy using just #vertices, #edges and diameter"
    if usingDefault:
        print "parameter not recognised, using default values"
    else:
        print "using: " + param + " " + str(v)
    print acc2
    print "*****************************************************"
    return (acc2, label)



def writeDEVToCSV(p):
    hdr=["Value", "Accuracy Score"]
    toFile=[]
    toFile.append(hdr)
    with open(data.datapath+"/ml_results/"+"dev_result_"+p+".csv","w") as outputFile:
        writer=csv.writer(outputFile)
        writer.writerow(hdr)
        for v in range(2, 25, 1):
            res = runMLFullMFD(v, p)
            val=[]
            val.append(res[1])
            val.append(res[0])
            writer.writerow(val)


def writeMFDToCSV(p):
    hdr=["Value", "Accuracy Score"]
    toFile=[]
    toFile.append(hdr)
    with open(data.datapath+"/ml_results/"+"mfd_result_"+p+".csv","w") as outputFile:
        writer=csv.writer(outputFile)
        writer.writerow(hdr)
        for v in range(2, 25, 1):
            res = runMLFullMFD(v, p)
            val=[]
            val.append(res[1])
            val.append(res[0])
            writer.writerow(val)

    #toFile.append(val)



writeMFDToCSV("max_leaf_nodes")
writeDEVToCSV("max_leaf_nodes")
writeMFDToCSV("max_depth")
writeDEVToCSV("max_depth")
writeMFDToCSV("min_samples_split")

    ##tree.export_graphviz(dt,out_file=data.datapath+"treeMFD.dot")
    # tree.export_graphviz(dt,out_file=data.datapath+"treeNoMFD.dot")
