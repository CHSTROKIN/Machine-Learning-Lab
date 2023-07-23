from math import log2
import operator
def calcShannonEnt(dataset):
    numEntries=len(dataset)#number of element inside the dataset
    labelCount={}
    for featvector in dataset:
        currentlabel=featvector[-1]
        if currentlabel not in labelCount.keys():
            labelCount[currentlabel]=0
        labelCount[currentlabel]+=1
    shannonEnt=0.0
    for key in labelCount.keys():
        prob=float((labelCount[key]))/float(numEntries)
        shannonEnt-=prob*log2(prob)
    return shannonEnt
def create_dataset():
    dataset=[
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    label=['input 1','input 2']
    return dataset,label
def splitDataset(dataset,axis,value):
    retdataset=[]
    return retdataset
def chooseBestFeatureToSplit(dataset):
    bestFeature = 0
    return bestFeature
def majorityCnt(classList): #usefull function, use it
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    myTree = None
    return myTree 

def classify(inputTree,featLabels,testVec):
    return Mone

if __name__=='__main__':
    dat,lab=create_dataset()
    print(calcShannonEnt(dat))
    print(chooseBestFeatureToSplit(dat))
    print('_______________________________________')
    print(splitDataset(dat,0,1))
    print(calcShannonEnt(splitDataset(dat,0,1)))
    print('_____________________________________')
    print(splitDataset(dat,0,0))
    print(calcShannonEnt(splitDataset(dat,0,0)))
    print('_____________________________________')    
    my_tree=createTree(dat,lab)
    print(my_tree)

