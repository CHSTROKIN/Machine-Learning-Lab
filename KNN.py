import numpy as np
from numpy.lib.function_base import diff
def create_dataset():
    dataset=np.array(
        [
        [1,2,3,4],
        [2,3,4,5],
        [4,112,234,423],
        [1234,234,234,32]
        ]
    )
    label=["inset","inset","mammal","mammal"]
    feature_name=["A","C","G","T"]
    return dataset,label,feature_name
def classify(inx=np.array,dataset=np.array,labels=[],k=int):
    '''
    inx is the input feature
    datset is the dataset:
        the format of dataset is 
        [
            fea1,
            fea2,
            fea3,
            fea4
            ......,
            fean
        ]
         there is no label in dataset
    lables is the label of according feature vector in the dataset
    '''
    datasetsize=dataset.shape[0]
    diffmat=np.tile(inx,(datasetsize,1))-dataset
    diffmat=diffmat*diffmat
    '''
    tile function will produce a matrix with (datasetsize,1) shape, and the element inside the matrix all will be inx
    [
        inx,
        inx,
        inx,
    ]
    '''
    sqdistance=diffmat.sum(axis=1)
    sorteddistance=sqdistance.argsort()
    classcount={}
    for i in range(k):
        votelable=labels[sorteddistance[i]]
        if(not (votelable in classcount.keys())):
            classcount[votelable]=0
        classcount[votelable]+=1
    tmax=-1
    ret=""
    for tkey in classcount.keys():
        val=classcount[tkey]
        if(val>tmax):
            ret=tkey 
            tmax=val 
    return ret
dataset,lab,fea=create_dataset()
inx=[113,213,312,431]
print(classify(inx,dataset,lab,2))
