import numpy as np

class PerceptronMulticapa:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1, num_epochs=1000):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # inicializar matrices de pesos
        self.w1 = np.random.randn(self.n_input, self.n_hidden)
        self.b1 = np.zeros((1, self.n_hidden))
        self.w2 = np.random.randn(self.n_hidden, self.n_output)
        self.b2 = np.zeros((1, self.n_output))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):
        # propagación hacia adelante
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, y_pred):
        # propagación hacia atrás
        error_output = y - y_pred
        delta_output = error_output * self.sigmoid_derivative(self.z2)

        error_hidden = np.dot(delta_output, self.w2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.z1)

        # actualizar pesos y sesgos
        self.w2 += self.learning_rate * np.dot(self.a1.T, delta_output)
        self.b2 += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.w1 += self.learning_rate * np.dot(X.T, delta_hidden)
        self.b1 += self.learning_rate * np.sum(delta_hidden, axis=0)

    def train(self, X_train, y_train):
        # entrenamiento
        for epoch in range(self.num_epochs):
            y_pred = self.forward(X_train)
            self.backward(X_train, y_train, y_pred)

    def predict(self, X_test):
        # predicción
        y_pred = self.forward(X_test)
        return np.round(y_pred)

