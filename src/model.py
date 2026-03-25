import numpy as np

def compute_mse(X, y, w):

    predictions = np.dot(X, w)
    errors = predictions - y
    mse = np.mean(np.square(errors))
    return mse

def compute_gradient(X, y, w):

    N = X.shape[0]
    predictions = np.dot(X, w)
    errors = predictions - y

    gradient = (2/N) * np.dot(X.T, errors)

    return gradient

def train_linear_regression(X, y, learning_rate=0.01, epochs=1000):

    n_features = X.shape[1]
    w = np.zeros(n_features)

    loss_history = []

    for i in range(epochs):
        gradient = compute_gradient(X, y, w)

        w = w - (learning_rate * gradient)

        current_loss = compute_mse(X, y, w)
        loss_history.append(current_loss)

        if i % 100 == 0:
            print(f"Epoch {i:4d} | MSE Loss: {current_loss:.4f}")
    
    print(f"Training complete. Final MSE Loss: {loss_history[-1]:.4f}")
    return w, loss_history