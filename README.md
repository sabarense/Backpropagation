# 🧠 Backpropagation para Funções Lógicas

Implementação do algoritmo **Backpropagation** para resolver funções lógicas **AND**, **OR** e **XOR**, com suporte para múltiplas entradas booleanas. Este projeto tem caráter educacional e visa ilustrar os fundamentos do aprendizado supervisionado com redes neurais multicamadas.

---

## 📋 Índice

- [📖 Descrição](#-descrição)
- [✨ Características](#-características)
- [🚀 Instalação](#-instalação)
- [💻 Como Usar](#-como-usar)
- [📊 Resultados](#-resultados)
- [🧠 Análise Teórica](#-análise-teórica)
- [📝 Licença](#-licença)

---

## 📖 Descrição

O **Backpropagation** é o algoritmo padrão para treinar redes neurais multicamadas ajustando pesos por meio do cálculo do gradiente do erro.

Este projeto demonstra:

- Como uma rede neural com camada oculta resolve funções lógicas **lineares** e **não lineares** (AND, OR e XOR)
- A importância da **taxa de aprendizado**, **bias** e **função de ativação** no aprendizado
- A evolução dos **pesos**, **bias** e acurácia durante o treinamento
- Visualização da fronteira de decisão para problemas com 2 entradas

---

## ✨ Características

- ✅ Suporte para **n entradas booleanas** (2 a 10)
- ✅ Implementação da rede neural com uma camada oculta
- ✅ Escolha da função lógica (**AND**, **OR**, **XOR**)
- ✅ Investigação da influência de parâmetros:
  - Taxa de aprendizado
  - Uso do bias
  - Funções de ativação (Sigmoide, Tangente Hiperbólica)
- ✅ Visualização gráfica da fronteira de decisão para 2 entradas
- ✅ Evolução dos pesos, bias e acurácia durante o treinamento

---

## 🚀 Instalação

### ✅ Pré-requisitos

- Python 3.7 ou superior

### 📦 Instale as dependências:

```bash
pip install numpy matplotlib
```

###🔄 Clonando o repositório:

git clone https://github.com/sabarense/Backpropagation.git

cd Backpropagation

pip install -r requirements.txt

###💻 Como Usar

```bash
python backpropagation.py
```

No terminal, você será solicitado a informar:

Qual função lógica deseja treinar (AND, OR, XOR)

Quantidade de entradas (mínimo 2, máximo 10)

Taxa de aprendizado

Uso ou não do bias

Função de ativação (sigmoid ou tanh)

Se forem escolhidas 2 entradas, será exibida uma visualização animada da evolução da fronteira de decisão.

### 📊 Resultados

🔹 Taxa de aprendizado

        - Taxas muito baixas tornam o treinamento lento e podem não convergir em poucas épocas.
        
        - Taxas muito altas causam instabilidade.
        
        - Taxa ideal encontrada: 0.1

🔹 Uso do bias

        - Essencial para o correto aprendizado de funções não linearmente separáveis, como o XOR.

🔹 Funções de ativação

        - Sigmoide: Tradicional, mas com saturação em valores extremos.
        - Tangente Hiperbólica (tanh): Melhor desempenho devido à saída centrada em zero.

## 🧠 Análise Teórica

O algoritmo Backpropagation ajusta os pesos da rede minimizando o erro da saída em relação à saída esperada, utilizando o gradiente descendente.

A atualização dos pesos segue a regra geral:

````
w ← w + η * δ * entrada

Onde:

η é a taxa de aprendizado

δ é o erro local calculado a partir da função de ativação e do erro da camada seguinte

````

### 📝 Licença
Este projeto é de uso educacional e está sob a licença MIT.
Sinta-se à vontade para estudar, modificar e reutilizar o código.