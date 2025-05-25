import numpy as np
import matplotlib.pyplot as plt


# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Função para gerar os dados de treinamento para AND, OR e XOR com n entradas
def generate_data(func, n):
    # Gera todas as combinações possíveis de entradas booleanas (0 ou 1)
    inputs = np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)

    # Aplica a função lógica a cada linha
    if func == 'AND':
        outputs = np.all(inputs == 1, axis=1).astype(int)
    elif func == 'OR':
        outputs = np.any(inputs == 1, axis=1).astype(int)
    elif func == 'XOR':
        # XOR para n bits: output 1 se o número de 1's for ímpar
        outputs = np.mod(np.sum(inputs, axis=1), 2)
    else:
        raise ValueError("Função lógica inválida.")

    return inputs, outputs.reshape(-1, 1)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate, activation_func, use_bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.use_bias = use_bias

        # Inicialização dos pesos
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, 1))

        if use_bias:
            self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
            self.b2 = np.random.uniform(-1, 1, (1, 1))
        else:
            self.b1 = np.zeros((1, hidden_size))
            self.b2 = np.zeros((1, 1))

        if activation_func == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_func == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Função de ativação inválida.")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        return self.a2

    def backward(self, X, y, output):
        error = y - output
        d_a2 = error * self.activation_derivative(self.z2)

        dW2 = np.dot(self.a1.T, d_a2)
        db2 = np.sum(d_a2, axis=0, keepdims=True)

        d_a1 = np.dot(d_a2, self.W2.T) * self.activation_derivative(self.z1)

        dW1 = np.dot(X.T, d_a1)
        db1 = np.sum(d_a1, axis=0, keepdims=True)

        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2
        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1

        return np.mean(error ** 2)

    def train(self, X, y, epochs=10000, verbose=False):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.backward(X, y, output)
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

            # Plotar fronteira a cada 1000 epochs se n=2
            if self.input_size == 2 and epoch % 1000 == 0:
                plot_decision_boundary(self, X, y, epoch)

    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)


def plot_decision_boundary(nn, X, y, epoch):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = nn.forward(grid)
    Z = (probs >= 0.5).reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    # Fundo colorido
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

    # Pontos de dados
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Paired, edgecolors='k', s=100, alpha=0.8)

    # Configurações visuais
    plt.title(f'Fronteira de decisão - Epoch {epoch}', fontsize=14)
    plt.xlabel('Entrada 1', fontsize=12)
    plt.ylabel('Entrada 2', fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("=== ALGORITMO BACKPROPAGATION - Funções Lógicas ===")
    func = input("Escolha a função lógica (AND, OR, XOR): ").upper()
    n = int(input("Número de entradas (2 a 10): "))
    while n < 2 or n > 10:
        n = int(input("Digite um número válido entre 2 e 10: "))

    learning_rate = float(input("Taxa de aprendizado (ex: 0.1): "))
    use_bias_input = input("Usar bias? (s/n): ").lower()
    use_bias = True if use_bias_input == 's' else False

    activation_func = input("Função de ativação (sigmoid, tanh): ").lower()
    while activation_func not in ['sigmoid', 'tanh']:
        activation_func = input("Digite sigmoid ou tanh: ").lower()

    X, y = generate_data(func, n)

    hidden_size = max(2, n // 2)
    nn = NeuralNetwork(input_size=n, hidden_size=hidden_size,
                       learning_rate=learning_rate,
                       activation_func=activation_func,
                       use_bias=use_bias)

    nn.train(X, y, epochs=10000, verbose=True)

    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y) * 100

    print(f"\nResultados para {func} com {n} entradas:")
    print("Entradas | Saída Esperada | Saída Predita")
    for i in range(len(X)):
        print(f"{X[i]} | {y[i][0]} | {predictions[i][0]}")
    print(f"\nAcurácia: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
