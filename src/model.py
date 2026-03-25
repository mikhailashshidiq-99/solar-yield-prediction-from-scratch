import numpy as np

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.loss_history = []

    def _compute_gradient(self, X, y):
        N = X.shape[0]
        predictions = np.dot(X, self.weights)
        errors = predictions - y
        gradient = (2 / N) * np.dot(X.T, errors)
        return gradient
    
    def _compute_mse(self, X,y):
        predictions = np.dot(X, self.weights)
        errors = predictions - y
        return np.mean(np.square(errors))

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.loss_history = []

        for i in range(slef.epochs):
            gradient = self._compute_gradient(X, y)
            self.weights -= (self.learning_rate * gradient)
            self.loss_history.append(self._compute_mse(X, y))

            if i % 100 == 0:
                print(f"Epoch {i:4d} | MSE Loss: {self.loss_history[-1]:.4f}")
                
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model DNE: Call .fit() before .predict()")

        return np.dot(X, self.weights)