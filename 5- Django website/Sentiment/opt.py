import numpy as np
class Adam():
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.999):
            self.alpha = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.v = []
            self.s = []

    def init_params(self,layers):
        self.v.clear()
        self.s.clear()
        for layer in layers:
            w = np.zeros_like(layer.W)
            b = np.zeros_like(layer.b)
            self.v.append([w,b])
            self.s.append([w,b])

    def update(self,layers,N):
        for i in range(len(layers)):
            self.v[i][0] = self.beta1 * self.v[i][0] + (1-self.beta1) * layers[i].dW
            self.v[i][1] = self.beta1 * self.v[i][1] + (1-self.beta1) * layers[i].db

            self.s[i][0] = self.beta2 * self.s[i][0] + (1-self.beta2) * np.square(layers[i].dW)
            self.s[i][1] = self.beta2 * self.s[i][1] + (1-self.beta2) * np.square(layers[i].db)

            deltaW = (-1 * self.alpha * self.v[i][0]) / (np.sqrt(self.s[i][0] + 0.001))
            deltab = (-1 * self.alpha * self.v[i][1]) / (np.sqrt(self.s[i][1] + 0.001))

            layers[i].W = layers[i].W + deltaW/N
            layers[i].b = layers[i].b + deltab/N