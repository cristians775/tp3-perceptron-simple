import random
import numpy as np
class Perceptron():
    def __init__(self, eta=0.1, itr=30000):
        self.eta = eta
        self.itr = itr
        
        
    def fit(self, X, y):
        #   length del conjunto de entrenamiento
        _len = len(X[0])
        #   creamos una lista de w con w0 incluido ej: w = [0, 0, 0] -> [w0, w1, w2]
        w = list(np.zeros(_len))
        # delta w
        _w = []
        error = 1
        i = 0
        while(error > 0 and i < self.itr ):
                random_index = random.randint(0, len(X)-1)
                X_random =X[random_index]
                Y_random =y[random_index]
                o = self.calculate_o(X_random, w)
                # Calculamos delta w -> (etha * error) * [w0, w1, w2]
                etha_error = self.eta * (Y_random-o)
                _w = [el * etha_error for i,el in enumerate(X_random)]
                # Calculamos el nuevo w -> w = [1, 1, 2] _w = [1, 3, 2]  w + _w = [2, 4, 6]
                w = [el + _w[i] for i,el in enumerate(w)]
                error = self.calculate_error(X,y,w)
                i +=1
                print("i", i)
    
        return w
                
       
    def calculate_o(self, X,w):
       
        # Calculamos h -> sumatoria wi * xi
        h = sum([(w[i] * x) for i, x in enumerate(X)])
        # Calculamos o
        o = 1 if h >= 0 else -1
        return o
    
    def calculate_error(self,X,y, w):
        error = 0
        for i in range(len(X)):
            h = sum([(w[i] * x) for i, x in enumerate(X[i])])
            o = 1 if h >=0 else -1
            error += abs(y[i] - o)
        return error