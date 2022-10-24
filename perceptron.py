import numpy as np
class Perceptron():
    def __init__(self, eta=0.1, itr=1000):
        self.eta = eta
        self.itr = itr
        
        
    def fit(self, X, y):
        #   length del conjunto de entrenamiento
        X_len = len(X[0])
        #   creamos una lista de w con w0 incluido ej: w = [0, 0, 0] -> [w0, w1, w2]
        w = list(np.zeros(X_len))
        # delta w
        _w = 0.0
        error = 1
        j = 0
        while(error>0 and j < self.itr ):
            for i in range(len(X)):
                o = self.predict(X[i], w)
                error = y[i]-o
                # Calculamos delta w -> (etha * error) * [w0, w1, w2]
                etha_error = self.eta * error
                _w = [el * etha_error for i,el in enumerate(X[i])]
                # Calculamos el nuevo w -> w = [1, 1, 2] _w = [1, 3, 2]  w + _w = [2, 4, 6]
                w = [el + _w[i] for i,el in enumerate(w)]
            j=+j
    
        return w
                
       
    def predict(self, X,w):
       
        # Calculamos h -> sumatoria wi * xi
        h = sum([(w[i] * x) for i, x in enumerate(X)])
        # Calculamos o
        o = 1 if h >0 else -1
        return o