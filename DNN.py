import numpy as np
from tqdm import tqdm
from tqdm import tqdm
from math import exp
from math import tanh
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
class InputLayer:
    def __init__(self, length):
        self.number = length
        self.element = np.ones(shape = (self.number, 1))


class DenseLayer:
    def __init__(self, neuron: int, activation: str) -> None:
        self.number = neuron
        self.activation = activation
        self.element = np.ones([0])
        self.init_success = False
        self.relu_k = 1
        self.tanh_k = 1
        self.b = np.ones([0])
        self.mem = np.ones([0])
        self.grads = np.zeros([0])  # dL/da
    def relu(self, result: np.array) -> np.array:
        relus = lambda x: self.relu_k * x if x > 0 else 0
        # print('relu inp'+str(result.shape))
        vrelus = np.vectorize(relus)
        return vrelus(result)

    def sigmoid(self, result: np.array) -> np.array:
        sigmoids = lambda x: 1 / (1 + exp(-x))
        vsigmoids = np.vectorize(sigmoids)
        return vsigmoids(result)
    def d_sigoid(self, result: np.array) -> np.array:
        return (sigmoid(result)*(1-sigmoid(result)))
    def forms(self, n: int) -> None:
        self.element = np.random.rand(self.number, n) # n is the number of neuron of next layer
        self.relu_k = 1
        self.b = np.random.rand(self.number, 1)
        # print("layer:", self.element.shape)
        self.init_success = True

    def forward(self, inp:np.array) -> np.array:
        if not self.init_success:
            raise Exception("ERROR:Layer not initialized")
        # print(inp.shape)
        # print(self.element.shape)
        tep = np.matmul(self.element, inp)
        tep += self.b
        # print('mul_res'+str(tep.shape))
        # remember the input value of matrix, so we can do backward propagation, it is called cache
        if self.activation == "relu":
            tep = self.relu(tep)
        if self.activation == "sigmoid":
            tep = self.sigmoid(tep)
        self.mem = inp 
        # print('ret:'+str(tep.shape))
        return tep

    def grad(self, grads: np.array, learning_rate: float):
        d_activation_function = np.ones(1)
        if(self.activation == "sigmoid"):
            d_activation_function = self.d_sigoid(grads)
        if(self.activation == "relu"):
            d_activation_function = np.array()
        # print('mem shape' + str(self.mem.shape))
        # print('activation sahpe'+ str(d_activation_function.shape))
        # print('grads shape'+ str(grads.shape))
        # print('bias shape' + str(self.b.shape))
        # print('weight shape'+str(self.element.shape))
        # print('memory shape'+str(self.mem.shape))
        d_weight = np.transpose(np.matmul(self.mem, np.transpose(d_activation_function)))
        # print('d weight shape::'+str(d_weight.shape))
        d_bias = (d_activation_function)
        self.b = self.b - learning_rate * d_bias
        self.element = self.element - learning_rate * d_weight
        # print('weight element shape' + str(self.element.shape))
        # print('grads shape' + str(grads.shape))
        d_next = np.matmul(np.transpose(self.element), grads)
        # print('next shape',d_next.shape) 
        # print('_______________________')

        return d_next

class Model:
    def __init__(self):
        self.layer_list = []
        self.loss = []

    def forms(self):
        pre_n = self.layer_list[0].number
        for i in range(1, len(self.layer_list)):
            self.layer_list[i].forms(pre_n)
            pre_n = self.layer_list[i].number

    def add(self, nlayer: DenseLayer) -> None:
        self.layer_list.append(nlayer)

    def forward(self, x) -> np.array:
        self.layer_list[0].element = np.array(x)
        temp = self.layer_list[0].element
        for i in range(1, len(self.layer_list)):
            temp = self.layer_list[i].forward(temp)
        return temp
    
    def mse(self,y:np.array, y_true:np.array) -> np.array:
        return np.sum((y - y_true)**2)
    
    def cross_entropy(y:np.array, y_true:np.array) -> np.array:
        return np.ones([1])#not implemented yet
    
    def delta_y(self,y, y_true) -> None:
        return 2*(y - y_true)
    def back(
        self,
        trainx: np.array,
        trainy: np.array,
        epoch: int,
        loss_function: str,
        learning_rate,
    ) -> None:
        if len(trainx) == 0:
            raise Exception("ERROR:TrianX is empty")
        if len(trainy) == 0:
            raise Exception("ERROR:TrianX is empty")
        if epoch <= 0:
            raise Exception("ERROR:Epoch should bigger than 0")
        if len(trainx[0]) != self.layer_list[0].number:
            raise Exception(
                "ERROR:The shape of the element of TrainX is different with InputLayer"
            )
        for i in tqdm(range(epoch)):
            for j in range(len(trainx)):
                x = trainx[j]
                print('xshape'+str(x.shape))
                predict = self.forward(x)
                if loss_function == "mse":
                    y = trainy[j]
                    y_loss = self.mse(predict, y)
                    self.loss.append(y_loss)
                    # print(y_loss)
                    dy = self.delta_y(predict, y)
                    # print("dy:", dy.shape)
                    for i in range(len(self.layer_list)-1):
                        cur = len(self.layer_list) - i - 1
                        layers = self.layer_list[cur]
                        # print('dy '+ str(dy.shape))
                        dy = layers.grad(dy, learning_rate)
def sigmoid(result: np.array) -> np.array:
    sigmoids = lambda x: 1 / (1 + exp(-x))
    vsigmoids = np.vectorize(sigmoids)
    return vsigmoids(result)

if __name__ == "__main__":
    m = Model()
    m.add(InputLayer(2))
    m.add(DenseLayer(10, "sigmoid"))
    m.add(DenseLayer(1, "sigmoid"))
    print("_______________")
    m.forms()
    print("_______________")
    print(np.transpose(np.expand_dims([109, 100], axis = 0)))
    print('forward')
    print(m.forward(np.transpose(np.expand_dims([10, 10], axis = 0))))
    print('end')
    trainx = np.array([np.transpose(np.array([1,1]))])
    trainy = np.array([[1]])
    print(trainy)
    print(trainx.shape)
    print(trainy.shape)
    m.back(trainx,trainy,2500,"mse",0.001)
    print(m.forward(np.transpose(np.expand_dims([10, 10], axis = 0))))
    plt.plot(m.loss)
    plt.show()