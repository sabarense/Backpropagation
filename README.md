# Backpropagation 

Implementação do algoritmo Backpropagation para AND, OR e XOR com **n entradas**.

## Funcionalidades
- Funções lógicas com qualquer número de entradas
- Investigação: taxa de aprendizado, bias, funções de ativação
- Visualizações e comparações

## Uso

Gerar dados
X, y = generate_logic_data(2, 'XOR')

## Gerar dados

Treinar rede
nn = NeuralNetwork(input_size=2, hidden_size=5)<br>
nn.train(X, y, epochs=1000)

## Testes
test_learning_rate_importance() # Taxa de aprendizado<br>
test_bias_importance() # Importância do bias<br>
test_activation_functions() # Sigmóide, Tanh, ReLU<br>
comprehensive_logic_test() # Teste completo<br>


## Parâmetros Investigados

| Item | Valores Testados |
|------|------------------|
| Taxa de Aprendizado | 0.01, 0.1, 0.5, 1.0, 2.0 |
| Bias | Com/Sem |
| Ativação | Sigmóide, Tanh, ReLU |

## Dependências
- pip install numpy matplotlib

## Resultados
- **Taxa ideal**: 0.1
- **Bias**: Essencial para XOR
- **Melhor ativação**: Tanh
