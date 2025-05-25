import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def generate_logic_data(n_inputs, logic_function):
    """Gera dados para funções lógicas com n entradas"""
    # Gerar todas as combinações possíveis de 0s e 1s
    X = np.array(list(product([0, 1], repeat=n_inputs)))

    if logic_function == 'AND':
        y = np.array([1 if all(row) else 0 for row in X])
    elif logic_function == 'OR':
        y = np.array([1 if any(row) else 0 for row in X])
    elif logic_function == 'XOR':
        y = np.array([1 if sum(row) % 2 == 1 else 0 for row in X])
    else:
        raise ValueError("Função lógica não suportada")

    return X, y


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.1, activation='sigmoid', use_bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.activation_name = activation

        # Inicialização dos pesos
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))

        if use_bias:
            self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
            self.b2 = np.random.uniform(-1, 1, (1, output_size))
        else:
            self.b1 = np.zeros((1, hidden_size))
            self.b2 = np.zeros((1, output_size))

        # Histórico de treinamento
        self.loss_history = []

        # Definir funções de ativação
        self._set_activation_functions()

    def _set_activation_functions(self):
        """Define as funções de ativação e suas derivadas"""
        if self.activation_name == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif self.activation_name == 'tanh':
            self.activation = self._tanh
            self.activation_derivative = self._tanh_derivative
        elif self.activation_name == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        else:
            raise ValueError("Função de ativação não suportada")

    def _sigmoid(self, x):
        """Função sigmóide"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x):
        """Derivada da função sigmóide"""
        s = self._sigmoid(x)
        return s * (1 - s)

    def _tanh(self, x):
        """Função tangente hiperbólica"""
        return np.tanh(x)

    def _tanh_derivative(self, x):
        """Derivada da tangente hiperbólica"""
        return 1 - np.tanh(x) ** 2

    def _relu(self, x):
        """Função ReLU"""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """Derivada da ReLU"""
        return (x > 0).astype(float)

    def forward(self, X):
        """Propagação para frente"""
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)

        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)  # Sempre sigmoid na saída

        return self.a2

    def backward(self, X, y, output):
        """Propagação para trás (Backpropagation)"""
        m = X.shape[0]

        # Gradientes da camada de saída
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Gradientes da camada oculta
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # Atualização dos pesos
        self.W2 -= self.learning_rate * dW2
        self.W1 -= self.learning_rate * dW1

        if self.use_bias:
            self.b2 -= self.learning_rate * db2
            self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000, verbose=False):
        """Treinamento da rede neural"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Calcular perda
            loss = np.mean((output - y) ** 2)
            self.loss_history.append(loss)

            # Backward pass
            self.backward(X, y, output)

            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}, Perda: {loss:.6f}")

        if verbose:
            print(f"Treinamento concluído. Perda final: {self.loss_history[-1]:.6f}")

    def predict(self, X):
        """Predição"""
        output = self.forward(X)
        return (output > 0.5).astype(int)


def test_activation_functions():
    """Testa diferentes funções de ativação"""
    print("=== TESTE DE FUNÇÕES DE ATIVAÇÃO ===\n")

    activations = ['sigmoid', 'tanh', 'relu']
    X, y = generate_logic_data(2, 'XOR')
    y = y.reshape(-1, 1)

    results = {}

    for activation in activations:
        print(f"\n--- Testando função {activation.upper()} ---")

        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                           learning_rate=0.5, activation=activation)

        nn.train(X, y, epochs=1000, verbose=True)

        predictions = nn.predict(X)
        accuracy = np.mean(predictions.flatten() == y.flatten())

        results[activation] = {
            'accuracy': accuracy,
            'final_loss': nn.loss_history[-1],
            'loss_history': nn.loss_history
        }

        print(f"Acurácia: {accuracy:.2%}")
        print("Predições vs Esperado:")
        for i in range(len(X)):
            print(f"  {X[i]} -> {predictions[i][0]} (esperado: {y[i][0]})")

    # Plotar comparação das perdas
    plt.figure(figsize=(12, 4))

    for i, activation in enumerate(activations):
        plt.subplot(1, 3, i + 1)
        plt.plot(results[activation]['loss_history'])
        plt.title(f'{activation.upper()}\nAcurácia: {results[activation]["accuracy"]:.2%}')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


def test_learning_rate_importance():
    """Investiga a importância da taxa de aprendizado"""
    print("\n=== IMPORTÂNCIA DA TAXA DE APRENDIZADO ===\n")

    learning_rates = [0.01, 0.1, 0.5, 1.0, 2.0]
    X, y = generate_logic_data(2, 'XOR')
    y = y.reshape(-1, 1)

    results = {}

    plt.figure(figsize=(15, 3))

    for i, lr in enumerate(learning_rates):
        print(f"\nTestando taxa de aprendizado: {lr}")

        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                           learning_rate=lr, activation='sigmoid')

        nn.train(X, y, epochs=1000, verbose=False)

        predictions = nn.predict(X)
        accuracy = np.mean(predictions.flatten() == y.flatten())

        results[lr] = {
            'accuracy': accuracy,
            'loss_history': nn.loss_history
        }

        print(f"Acurácia final: {accuracy:.2%}")

        # Plotar curva de perda
        plt.subplot(1, 5, i + 1)
        plt.plot(nn.loss_history)
        plt.title(f'LR = {lr}\nAcc: {accuracy:.1%}')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


def test_bias_importance():
    """Investiga a importância do bias"""
    print("\n=== IMPORTÂNCIA DO BIAS ===\n")

    X, y = generate_logic_data(2, 'XOR')
    y = y.reshape(-1, 1)

    configurations = [
        ('Com Bias', True),
        ('Sem Bias', False)
    ]

    results = {}

    plt.figure(figsize=(10, 4))

    for i, (name, use_bias) in enumerate(configurations):
        print(f"\nTestando: {name}")

        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                           learning_rate=0.5, activation='sigmoid',
                           use_bias=use_bias)

        nn.train(X, y, epochs=1000, verbose=True)

        predictions = nn.predict(X)
        accuracy = np.mean(predictions.flatten() == y.flatten())

        results[name] = {
            'accuracy': accuracy,
            'loss_history': nn.loss_history
        }

        print(f"Acurácia: {accuracy:.2%}")

        # Plotar curva de perda
        plt.subplot(1, 2, i + 1)
        plt.plot(nn.loss_history)
        plt.title(f'{name}\nAcurácia: {accuracy:.2%}')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


def comprehensive_logic_test():
    """Teste abrangente das funções lógicas com Backpropagation"""
    print("\n=== TESTE ABRANGENTE - BACKPROPAGATION ===\n")

    logic_functions = ['AND', 'OR', 'XOR']
    n_inputs_list = [2, 3, 4]

    for logic_func in logic_functions:
        print(f"\n--- Função {logic_func} ---")

        for n_inputs in n_inputs_list:
            print(f"\nTestando {logic_func} com {n_inputs} entradas:")

            X, y = generate_logic_data(n_inputs, logic_func)
            y = y.reshape(-1, 1)

            # Ajustar tamanho da camada oculta baseado no número de entradas
            hidden_size = max(4, n_inputs * 2)

            nn = NeuralNetwork(input_size=n_inputs, hidden_size=hidden_size,
                               output_size=1, learning_rate=0.5,
                               activation='sigmoid')

            nn.train(X, y, epochs=1000, verbose=False)

            predictions = nn.predict(X)
            accuracy = np.mean(predictions.flatten() == y.flatten())

            print(f"  Acurácia: {accuracy:.2%}")
            print(f"  Perda final: {nn.loss_history[-1]:.6f}")

            # Mostrar algumas predições para verificação
            if n_inputs == 2:
                print("  Verificação completa:")
                for i in range(len(X)):
                    print(f"    {X[i]} -> {predictions[i][0]} (esperado: {y[i][0]})")


if __name__ == "__main__":
    # Executar todos os testes
    test_activation_functions()
    test_learning_rate_importance()
    test_bias_importance()
    comprehensive_logic_test()
