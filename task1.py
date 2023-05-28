from __future__ import annotations

import numpy as np
from numpy import ndarray
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def mae(preds: ndarray, actuals: ndarray):
    return np.mean(np.abs(preds - actuals))


def rmse(preds: ndarray, actuals: ndarray):
    return np.sqrt(np.mean(np.power(preds - actuals, 2)))


def init_weights(input_size: int, hidden_size: int) -> dict[str, ndarray]:
    weights: dict[str, ndarray] = {}
    weights['W1'] = np.random.randn(input_size, hidden_size)
    weights['B1'] = np.random.randn(1, hidden_size)
    weights['W2'] = np.random.randn(hidden_size, 1)
    weights['B2'] = np.random.randn(1, 1)
    return weights


Batch = tuple[ndarray, ndarray]


def generate_batch(X: ndarray, y: ndarray, start: int = 0, batch_size: int = 10) -> Batch:
    assert X.ndim == y.ndim == 2, "X and Y must be 2 dimensional"
    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start
    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
    return X_batch, y_batch


def permute_data(X: ndarray, y: ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-1.0 * x))


def sigmoid_der(x: ndarray) -> ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(data: ndarray) -> ndarray:
    c = data.copy()
    c[c <= 0] = 0
    return c


def relu_der(data: ndarray) -> ndarray:
    c = data.copy()
    c[c <= 0] = 0
    c[c > 0] = 1
    return c


def activation(data: ndarray) -> ndarray:
    return sigmoid(data)


def activation_der(data: ndarray) -> ndarray:
    return sigmoid_der(data)


def forward_loss(X: ndarray, y: ndarray, weights: dict[str, ndarray]) -> tuple[ndarray, dict[str, ndarray]]:
    M1 = np.dot(X, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = activation(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']
    loss = np.mean(np.power(y - P, 2))
    forward_info: dict[str, ndarray] = {}
    forward_info['X'] = X
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = P
    forward_info['y'] = y
    return loss, forward_info


def loss_gradients(forward_info: dict[str, ndarray], weights: dict[str, ndarray]) -> dict[str, ndarray]:
    dLdP = -2 * (forward_info['y'] - forward_info['P'])
    dPdM2 = np.ones_like(forward_info['M2'])
    dLdM2 = dLdP * dPdM2
    dPdB2 = np.ones_like(weights['B2'])
    dLdB2 = (dLdP * dPdB2).sum(axis=0)
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))
    dLdW2 = np.dot(dM2dW2, dLdP)
    dM2dO1 = np.transpose(weights['W2'], (1, 0))
    dLdO1 = np.dot(dLdM2, dM2dO1)
    dO1dN1 = activation_der(forward_info['N1'])
    dLdN1 = dLdO1 * dO1dN1
    dN1dB1 = np.ones_like(weights['B1'])
    dN1dM1 = np.ones_like(forward_info['M1'])
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
    dLdM1 = dLdN1 * dN1dM1
    dM1dW1 = np.transpose(forward_info['X'], (1, 0))
    dLdW1 = np.dot(dM1dW1, dLdM1)
    loss_gradients: dict[str, ndarray] = {}
    loss_gradients['W2'] = dLdW2
    loss_gradients['B2'] = dLdB2.sum(axis=0)
    loss_gradients['W1'] = dLdW1
    loss_gradients['B1'] = dLdB1.sum(axis=0)
    return loss_gradients


def train(X: ndarray, y: ndarray, n_iter: int = 1000, learning_rate: float = 0.1, batch_size: int = 100,
          return_losses: bool = False, return_weights: bool = False, seed: int = 1) -> tuple[list[ndarray], dict[str, ndarray]] | None:
    if seed:
        np.random.seed(seed)
    start = 0
    weights = init_weights(X.shape[1], 2)
    R: dict[str, ndarray] = {}
    R['W1'] = np.zeros_like(weights['W1'])
    R['B1'] = np.zeros_like(weights['B1'])
    R['W2'] = np.zeros_like(weights['W2'])
    R['B2'] = np.zeros_like(weights['B2'])
    M: dict[str, ndarray] = {}
    M['W1'] = np.zeros_like(weights['W1'])
    M['B1'] = np.zeros_like(weights['B1'])
    M['W2'] = np.zeros_like(weights['W2'])
    M['B2'] = np.zeros_like(weights['B2'])
    e0 = 1e-6
    b1 = 0.8
    b2 = 0.88
    X, y = permute_data(X, y)
    if return_losses:
        losses = []
    for i in range(n_iter):
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0
        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size
        loss, forward_info = forward_loss(X_batch, y_batch, weights)
        if return_losses:
            losses.append(loss)
        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            M[key] = b1 * M[key] + (1 - b1) * loss_grads[key]
            R[key] = b2 * R[key] + (1 - b2) * np.power(loss_grads[key], 2)
            weights[key] -= (M[key] / (np.sqrt(R[key] + e0))) * learning_rate
    if return_weights:
        return losses, weights
    return None


def predict(X: ndarray, weights: dict[str, ndarray]) -> ndarray:
    M1 = np.dot(X, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = activation(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']
    return P


X, y, c = make_regression(n_samples=100, n_features=7, n_informative=2, noise=3, coef=True)
y = y.reshape((-1, 1))
X_train, y_train = X[:70], y[:70]
X_test, y_test = X[70:], y[70:]
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [0], [0], [1]])
X_test, y_test = X_train, y_train
losses, weights = train(X_train, y_train,
                        n_iter=1000,
                        learning_rate=0.1,
                        batch_size=1,
                        return_losses=True,
                        return_weights=True)
print(*losses[-5:])
preds = predict(X_train, weights)
print(preds)
preds = predict(X_test, weights)
print(preds)
print("Mean absolute error:", round(mae(preds, y_test), 4), "\nRoot mean squared error:",
      round(rmse(preds, y_test), 4))
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
print(preds)
print("Mean absolute error:", round(mae(preds, y_test), 4), "\nRoot mean squared error:",
      round(rmse(preds, y_test), 4))
print(np.round(weights['W1'].reshape(-1), 4))
print(np.round(weights['W2'].reshape(-1), 4))
print(np.round(lr.coef_, 4))
print(np.round(weights['B1'], 4))
print(np.round(weights['B2'], 4))
print(np.round(lr.intercept_, 4))
print("True coef:", c)
