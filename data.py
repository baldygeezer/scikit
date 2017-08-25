from numpy import loadtxt

datapath = "data/"


def processData(filename):
    osmData = loadtxt(datapath + filename, delimiter=',')
    tTargets = osmData[:, 0]
    tFeatures = osmData[:, 1:]
    #data={tTargets,tFeatures}
    ##print osmData.shape
   # print tFeatures[0]
    #print tFeatures
    return tTargets,tFeatures


def processDataNoMFD(filename):
    osmData = loadtxt(datapath + filename, delimiter=',')
    tTargets = osmData[:, 0]
    tFeatures = osmData[:, 17:]
    #data={tTargets,tFeatures}
    ##print osmData.shape
   # print tFeatures[0]
    #print tFeatures
    return tTargets,tFeatures