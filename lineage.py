import numpy as np
import random
import math
'''
第一次用pynum写线性 回归的框架
'''
class neuro:
    size_of_one_traning_example=0
    learning_rate=0
    the_train_set_x=[]
    the_train_set_y=[]
    the_weight=[]
    the_constant=0
    def __init__(self,learn,size_examples,train_value_x,trian_value_y):#定义
        self.learning_rate=learn      #                                              #学习率
        self.size_of_one_traning_example=size_examples                              #一个样本的大小
        self.the_train_set_x=train_value_x                                          #输入样本
        self.the_train_set_y=trian_value_y                                          #正确答案样本
        self.the_weight=np.random.rand(self.size_of_one_traning_example)#和一个样本的大小一致
    def random_weight(self):#随机权重
        self.the_weight=np.random.rand(self.size_of_one_traning_example)#和一个样本的大小一致
        return 
    def print_weight(self):
        print(self.the_weight,' ',self.the_constant)
    def segmoid(self,n):#激活函数
        return 1/(1+math.exp(-n))
    def get_input(self,the_input_list=[]):#更改训练集
        self.the_train_set=the_input_list
        return
    def train(self,train_cycle):#用梯度下降法
        for i in range(0,train_cycle):#训练次数
            for j in range(0,len(self.the_train_set_x)):#遍历样本集合
                one_train_example=self.the_train_set_x[j]#提取一个样本
                one_correct_answer=self.the_train_set_y[j]#提取正确答案
                the_ret_value=self.get_value(one_train_example)#通过输入算出答案
                mis=the_ret_value-one_correct_answer#误差
                for k in range(0,len(self.the_weight)):#遍历权重集合
                    self.the_weight[k]=self.the_weight[k]-self.learning_rate*(mis)*one_train_example[k]*2#梯度下降,在这里用一个数字，因为维度是1
                    self.the_constant=self.the_constant-self.learning_rate*(mis)*2#梯度下降B
        return
    def get_value(self,one_input_value):#返回lineage回归的输出
        z=np.dot(self.the_weight,one_input_value)
        a=np.sum(z)
        a+=self.the_constant
        return a
def func(n):
    result=[]
    i=2
    end=0
    while end<n:
        if(isprime(i)):
            result.append(i)
            end+=1
        i+=1
    return result
def isprime(n):
    for i in range(2,int(math.sqrt(n)+1)):
         if(n%i==0):
            return False
    return True
def func2(n):
    return 2*n+1
if __name__ == '__main__':
    print('线性回归已经开始运行')
    x=[]
    y=[]
    for i in range(0,100):
        y.append(2*i+1)
        value=[]
        value.append(i)
        x.append(value)
       # print(x[i],' ',y[i])
    my_neuro=neuro(0.0001,1,x,y)
    my_neuro.print_weight()
    my_neuro.train(10000)
    print('____________________')
    my_neuro.print_weight()
    print(my_neuro.get_value(1000))
    

    
