```{python}
import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.L = len(layer_sizes) - 1  # number of layers (excluding input)
        self.learning_rate = learning_rate
        self.W = {}
        self.b = {}
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2. / layer_sizes[l-1])
            self.b[l] = np.zeros((layer_sizes[l], 1))

    def forward(self, x):
        a = {0: x}
        z = {}
        for l in range(1, self.L):
            z[l] = self.W[l] @ a[l-1] + self.b[l]
            a[l] = relu(z[l])
        z[self.L] = self.W[self.L] @ a[self.L - 1] + self.b[self.L]
        a[self.L] = softmax(z[self.L])
        return a, z

    def backward(self, x, y, a, z):
        m = x.shape[1]
        grads = {}
        delta = a[self.L] - y  # Output error

        for l in reversed(range(1, self.L + 1)):
            grads[f'dW{l}'] = (delta @ a[l - 1].T) / m
            grads[f'db{l}'] = np.sum(delta, axis=1, keepdims=True) / m
            if l > 1:
                delta = (self.W[l].T @ delta) * relu_derivative(z[l - 1])
        return grads

    def update_params(self, grads):
        for l in range(1, self.L + 1):
            self.W[l] -= self.learning_rate * grads[f'dW{l}']
            self.b[l] -= self.learning_rate * grads[f'db{l}']

    def train_batch(self, X, Y):
        a, z = self.forward(X)
        loss = cross_entropy(a[self.L], Y)
        grads = self.backward(X, Y, a, z)
        self.update_params(grads)
        return loss

    def predict(self, X):
        a, _ = self.forward(X)
        return np.argmax(a[self.L], axis=0)
    

if __name__ == '__main__':

    X_xor = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]])  # shape (2, 4)
    y_xor = np.array([0, 1, 1, 0])    # shape (4,)
    Y_xor = one_hot(y_xor, num_classes=2)  # shape (2, 4)

    # Define a small network: 2 input, 2 hidden, 2 output (for one-hot)
    xor_layer_sizes = [2, 2, 2]
    xor_nn = NeuralNetwork(xor_layer_sizes, learning_rate=0.1)

    # Train on XOR
    xor_epochs = 5000
    for epoch in range(xor_epochs):
        loss = xor_nn.train_batch(X_xor, Y_xor)
        if (epoch + 1) % 500 == 0 or epoch == 0:
            preds = xor_nn.predict(X_xor)
            acc = np.mean(preds == y_xor)
            print(f"[XOR] Epoch {epoch + 1:04d}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    # Test predictions
    xor_preds = xor_nn.predict(X_xor)
    print("[XOR] Predictions:", xor_preds)
    print("[XOR] Ground truth:", y_xor)
```

Let $\mathcal{N}$ be a neural network with $L$ layers indexed by $l$. Each layer has $n^{[l]}$ neurons. Also let $x_0$ is input and $x_{L - 1}$ is the output, each layer has input $x_l$ and output $x_{l + 1}$, then $\forall l > 0, \exists W^{[l]} \in \mathcal{M}_{n^{[l]} \times n^{[l - 1]}}(\mathbb{R}), b^{[l]} \in \mathbb{R}, \sigma^{[l]}: \mathbb{R}^l \times \to \mathbb{R}^l$ s.t $\mathbf{x}_{l + 1} = \sigma^{[l + 1]}(W^{[l + 1]}(\mathbf{x}_l) + b^{[l + 1]})$. Therefore, $\mathbf{x}_{L - 1} = \sigma^{[L - 1]}(W^{[L - 1]}( \sigma^{[L - 2]}(W^{[L - 2]}( \dots \sigma^{[1]}(W^{[1]} + b^{[1]}) \dots ) + b^{[L - 2]})) + b^{[L - 1]})$. Let the target result be $x$, then we can have loss function $C = \frac{1}{2}(\mathbf{x} - \mathbf{x}_L)^2$. Then $\forall l \in \mathbb{N}^*_{<L}, \forall j \leq n_l, i \leq n_{[l - 1]}, \frac{\partial C}{\partial w^{[l]}} \in \mathbb{R} \land \frac{\partial C}{\partial b^{[l]}} \in \mathbb{R}$. Then each weight and bias can be adjust based on the partial differential and learning rate $\eta$. Repeat for multiple times, the loss $C$ will reach the minima.

$\frac{\partial C}{\partial w^{[k]}_{ij}} = \frac{\partial C}{\partial \textbf{x}_{k + 1}} \frac{\partial \mathbf{x}_{k + 1}}{\partial w^{[k]}_{ij}} = \frac{\partial C}{\partial \mathbf{x}_{k + 1}} \frac{\partial \mathbf{x}_{k + 1}}{\partial W^{[k + 1]}(\mathbf{x}_k)} \frac{\partial W^{[k + 1]}(\mathbf{x}_k)}{\partial w^{[k]}_{ij}} = \frac{\partial C}{\mathbf{x}_{k + 1}} \sigma(W^{[k + 1]}(\mathbf{x}_{k + 1}))(1 - \sigma(W^{[k + 1]}(\mathbf{x}_{k + 1})))x^{[k]}_j$

$\frac{\partial C}{\partial \mathbf{x}_{k + 1}} = \begin{cases}\mathbf{x} - \mathbf{x}_L \text{ , } k = L - 1 \\
(W^{[k + 1]})^T \circ \frac{\partial C}{\partial \mathbf{x}_{k + 2}} \circ (\sigma^{[k + 2]})^{'} \circ W^{[k + 2]}(\mathbf{x}_{k + 1}) \text{ , otherwise} 
\end{cases}$