import numpy as np
import time

class NeuralNetwork:
    def __init__(self):

        self.wi = np.random.randn(50, 60) * np.sqrt(2/60)
        self.bi = np.zeros((50, 1))
        self.wh = np.random.randn(50, 50) * np.sqrt(2/50)
        self.bh = np.zeros((50, 1))
        self.wo = np.random.randn(40, 50) * np.sqrt(2/50)
        self.bo = np.zeros((40, 1))

    def ReLU(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = x.reshape(-1, 1)

        self.h1 = self.ReLU(self.wi @ x + self.bi)
        self.h2 = self.ReLU(self.wh @ self.h1 + self.bh)
        self.o = self.wo @ self.h2 + self.bo

        return self.o

    def backward(self, x, y, learning_rate):
        y_pred = self.forward(x)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        dL_do = 2 * (y_pred - y)
        dwo = dL_do @ self.h2.T
        dbo = dL_do

        dL_dh2 = (self.wo.T @ dL_do)*(self.h2 > 0)
        dwh = dL_dh2 @ self.h1.T
        dbh = dL_dh2

        dL_dh1 = (self.wh.T @ dL_dh2)*(self.h1 > 0)
        dwi = dL_dh1 @ x.T
        dbi = dL_dh1

        self.wo -= learning_rate * dwo
        self.bo -= learning_rate * dbo
        self.wh -= learning_rate * dwh
        self.bh -= learning_rate * dbh
        self.wi -= learning_rate * dwi
        self.bi -= learning_rate * dbi

    def train(self, objects, answer, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(objects)-99):
                x = objects[i]
                y = answer[i]

                y_pred = self.forward(x)

                loss = np.mean((y_pred - y.reshape(-1, 1))**2)
                total_loss += loss

                self.backward(x, y, learning_rate)
            print(f"Epoch {epoch+1}/{epoch}, Loss: {total_loss/len(objects):.4f}")

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    data = np.loadtxt('Dataset10000.txt')

    objects = data[:, :60]
    answer = data[:, 60:]

    nn = NeuralNetwork()
    start = time.time()
    nn.train(objects, answer, epochs=10, learning_rate=0.001)
    print(start - time.time())

    # test o model
    datatest = []
    for i in range(100):
        x = objects[-(i+1)]
        y = answer[-(i+1)]

        y_pred = nn.forward(x)

        loss = np.mean(abs(y_pred - y.reshape(-1, 1)))
        datatest.append(loss)

    print(np.mean(datatest))

    # result
    x = objects[-5]     #type any data
    y = answer[-5]

    y_pred = nn.forward(x)

    print(f"y real: {y}")
    print(f"y pred: {y_pred}")
    print(f"mistake: {np.mean(abs(y_pred - y.reshape(-1, 1)))}")