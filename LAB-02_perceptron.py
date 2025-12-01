import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                err = yi - pred
                self.weights += self.lr * err * xi
                self.bias += self.lr * err

    def predict(self, x):
        return 1 if (np.dot(x, self.weights) + self.bias) >= 0 else 0

# Training data
X_train = np.array([[2,3], [1,4], [3,5], [4,2]])
y_train = np.array([0,0,1,1])

# Train
p = Perceptron(lr=0.01, epochs=100)
p.fit(X_train, y_train)

# Predict
new_point = np.array([2,4])
print("Prediction:", p.predict(new_point))
