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
    for fv in dataset:
        if fv[axis]==value:
            reducedFeatvec=fv[:axis]
            reducedFeatvec.extend(fv[axis+1:])
            retdataset.append(reducedFeatvec)
    return retdataset
def chooseBestFeatureToSplit(dataset):
    numFeature=len(dataset[0])-1
    baseEntropy=calcShannonEnt(dataset)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(0,numFeature):
        featList=[example[i] for example in dataset]#the ith column in the dataset
        uniqueVals=set(featList)#the unique value of the ith column
        newEntropy=0.0
        for value in uniqueVals:
            subdataset=splitDataset(dataset,i,value)
            prob=len(subdataset)/float((len(dataset)))# what probality
            newEntropy+=prob*calcShannonEnt(subdataset)#because each new dataset only take small amount of the whole dataset
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i 
        return bestFeature
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataSet, bestFeat, value),subLabels)
    return myTree 

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

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

