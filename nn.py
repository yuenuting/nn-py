#author: NuTing-YUE(21896195@qq.com)  DaJiong-YUE(13258932@qq.com)    2018-10-06
import numpy as np

class Layer:
    def __init__(self, input_size, output_size, weight_scale):
        self.W = (2*np.random.random((input_size, output_size)) -1.0) * weight_scale  #y=x1^2+x2^2: 0.9
        self.b = np.zeros((output_size))

    def recept(self, I):
        O = self.__activation(np.dot(I, self.W)+self.b)
        return O

    def adapt(self, Distance, O, I, learning_rate=0.008):  #learning_rate:  y=x1^2+x2^2: 0.001    #(x1^5 + x2^3) * exp(-x1^2 - x2^2)
        Gradient = self.__activation_derivative(O)
        Delta = Distance * Gradient
        self.W += learning_rate * I.T.dot(Delta)
        self.b += learning_rate * np.sum(Delta)
        return Delta

    def __activation(self, X):
        return np.tanh(X)  #sigmoid: 1/(1 + np.exp(-X)

    def __activation_derivative(self, X):
        return 1.0-np.tanh(X)**2  #sigmoid-derivative: X * (1-X)

class Network:
    def __init__(self, input_size, hidden_size, output_size, research_unfreeze_weight_epoch):
        self.i2h = Layer(input_size, hidden_size, 0.01)
        self.h2o = Layer(hidden_size, output_size, 0.9)
        #
        self.research_unfreeze_weight_epoch = research_unfreeze_weight_epoch

    def train(self, X, Y, research_epoch):
        if research_epoch == self.research_unfreeze_weight_epoch:
            self.i2h.W = (2*np.random.random((self.i2h.W.shape[0], self.i2h.W.shape[1])) -1.0) * 0.9
        #
        H = self.i2h.recept(X)
        O = self.h2o.recept(H)
        DistanceO = Y - O
        DeltaO = self.h2o.adapt(DistanceO, O, H)
        #
        if research_epoch >= self.research_unfreeze_weight_epoch:
            DistanceH = DeltaO.dot(self.h2o.W.T)
            DeltaH = self.i2h.adapt(DistanceH, H, X)

    def loss(self, O, Y):
        if Y is not None:
            abs_loss = np.sum(np.abs(Y - O)) / O.shape[0]
            mse_loss = 0.5 * (np.sum(np.power(Y-O,2)))
            cross_entropy_loss = -np.mean(np.log(np.clip(O, 1e-12, 1.0-1e-12)) * Y)
            print("abs_loss=%.6f"%abs_loss, "    ", "mse_loss=%.6f"%mse_loss, "    ","cross_entropy_loss=%.6f"%cross_entropy_loss)

    def inference(self, X):
        H = self.i2h.recept(X)
        O = self.h2o.recept(H)
        return O

class Dataset:
    def __init__(self, grid_size, add_one=True):
        d1 = np.linspace(-0.7, +0.7, grid_size)
        d2 = np.linspace(-0.7, +0.7, grid_size)
        D1, D2 = np.meshgrid(d1, d2)
        X = []
        Y = []
        for i in range(grid_size*grid_size):
            i1 = np.random.choice(D1.shape[1], 1, replace=False)[0]
            i2 = np.random.choice(D2.shape[0], 1, replace=False)[0]
            x1 = D1[0][i1]
            x2 = D2[i2][0]            
            y = (x1**5 + x2**3) * np.exp(-x1**2 - x2**2)  #y =  x1**2 + x2**2
            if add_one:           
                X.append([x1, x2, 1])  #[ , , 1] #CONCAT-ONE: Input CONCAT-ONE/AVOID-ZERO，optimize distribution of Output(not near to 0), make problem can be convergenced。(high-dimension maybe not sensitive)
            else:
                X.append([x1, x2])
            Y.append([y])
        self.X = np.array(X)
        self.Y = np.array(Y)

    def mini_batch(self, mini_batch=1):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        start_index = 0
        excerpt = indices[start_index: start_index + mini_batch]
        return self.X[excerpt], self.Y[excerpt]
   
if __name__ == '__main__':
    research_unfreeze_weight_epoch_list = [0, 1, 2, 3, 4, 5]
    for research_unfreeze_weight_epoch in research_unfreeze_weight_epoch_list:
        print('\nresearch_unfreeze_weight_epoch=%d'%research_unfreeze_weight_epoch)
        np.random.seed(122333)
        train = Dataset(30)
        test = Dataset(20)   #i.i.d
        wave = Network(train.X.shape[1], 9, train.Y.shape[1], research_unfreeze_weight_epoch)
        for i in range(100*5):
            if i % (50) == 0:
                X, Y = test.mini_batch(mini_batch=2)
                O = wave.inference(X)
                wave.loss(O, Y)
            X, Y= train.mini_batch(mini_batch=3)
            wave.train(X, Y, i)
