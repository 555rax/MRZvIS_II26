import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
  return x * (1 - x)


class MultiLayerPerceptron:

  def __init__(self, input_size=2, hidden_size=2, output_size=1, lr=0.1):
    np.random.seed(42)
    self.W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
    self.b1 = np.zeros((1, hidden_size))
    self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
    self.b2 = np.zeros((1, output_size))
    self.lr = lr
    self.errors = []

  def train(self, X, y_target, epochs=5000, error_threshold=0.01):
    for epoch in range(epochs):
      hidden_input = np.dot(X, self.W1) + self.b1
      hidden_output = sigmoid(hidden_input)

      final_input = np.dot(hidden_output, self.W2) + self.b2
      final_output = sigmoid(final_input)

      error = y_target - final_output
      mse = np.mean(error**2)
      self.errors.append(mse)

      d_output = error * sigmoid_derivative(final_output)
      d_hidden = d_output.dot(self.W2.T) * sigmoid_derivative(hidden_output)

      self.W2 += hidden_output.T.dot(d_output) * self.lr
      self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
      self.W1 += X.T.dot(d_hidden) * self.lr
      self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

      if mse < error_threshold:
        print(
            f"MLP: Convergencia alcanzada en la época {epoch},"
            f" error={mse:.4f}"
        )
        break

    return final_output

  def predict(self, x):
    hidden_input = np.dot(x, self.W1) + self.b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, self.W2) + self.b2
    return sigmoid(final_input)


class SingleLayerPerceptron:

  def __init__(self, input_size=2, output_size=1, lr=0.1):
    self.W = np.random.uniform(-0.5, 0.5, (input_size, output_size))
    self.b = np.zeros((1, output_size))
    self.lr = lr
    self.errors = []

  def train(self, X, y_target, epochs=2000):
    for epoch in range(epochs):
      net_input = np.dot(X, self.W) + self.b
      output = sigmoid(net_input)

      error = y_target - output
      mse = np.mean(error**2)
      self.errors.append(mse)

      d_output = error * sigmoid_derivative(output)

      self.W += X.T.dot(d_output) * self.lr
      self.b += np.sum(d_output, axis=0, keepdims=True) * self.lr

    return output


def main():
  X = np.array([[0, 0], [0, -6], [-6, 0], [-6, -6]])
  y = np.array([[0], [-6], [-6], [0]])

  y_min, y_max = y.min(), y.max()
  y_norm = (y - y_min) / (y_max - y_min)

  mlp = MultiLayerPerceptron()
  mlp_final_output = mlp.train(X, y_norm)
  mlp_output_rescaled = mlp_final_output * (y_max - y_min) + y_min

  print("\n--- Resultados del entrenamiento MLP ---")
  for i in range(len(X)):
    print(f"Entrada: {X[i]} → Salida: {mlp_output_rescaled[i][0]:.2f}")

  slp = SingleLayerPerceptron()
  slp_final_output = slp.train(X, y_norm)
  slp_output_rescaled = slp_final_output * (y_max - y_min) + y_min

  print("\n--- Resultados del entrenamiento del Perceptrón Simple ---")
  for i in range(len(X)):
    print(f"Entrada: {X[i]} → Salida: {slp_output_rescaled[i][0]:.2f}")

  print("\n--- Verificación del funcionamiento de la red ---")
  test_inputs = [[0, -6], [-6, -6], [-3, -6], [-6, -2]]
  for inp in test_inputs:
    norm_res = mlp.predict(np.array([inp]))
    result = norm_res.item() * (y_max - y_min) + y_min
    print(f"Entrada: {inp} → Salida de la red: {result:.2f}")

  plt.plot(mlp.errors, label="MLP")
  plt.plot(slp.errors, label="SLP")
  plt.xlabel("Época")
  plt.ylabel("Error (MSE)")
  plt.title("Comparación de errores entre MLP y SLP")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()