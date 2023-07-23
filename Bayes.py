import numpy as np
def loadDataSet():
    postingList=[
        ['my','dog','has','flea','help'],
        ['you','are','stupid'],
        ['go','to','hell'],
        ['the','dog','is','cute'],
        ['quit','this','forum']
    ]
    classvec=[0,1,1,0,1]
    return postingList,classvec
def createvocablist(dataset):
    vocab=set([])
    for docu in dataset:
        vocab=vocab|set(docu)
    return list(vocab)
def setofword2vec(vocab,inputset):
    revec=[0]*len(vocab)
    for word in inputset:
        revec[vocab.index(word)]=1
    return revec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pabusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.zeros(numWords)
    p1Num=np.zeros(numWords)
    p0Denom=0.0
    p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix)
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1vect=p1Num/p1Denom
    p0vect=p0Num/p0Denom
    return p0vect,p1vect,pabusive
