import numpy as np
import random

class NeuralNetwork:
  def __init__(self, input_nodes, hidden_nodes, output_nodes):
    self.weights_ih = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
    self.weights_ho = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
    self.bias_hidden = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, 1))
    self.bias_output = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, 1))

  def activation_function(self, name, x):
    if name == "sigmoid":
      return 1 / (1 + np.exp(-x))

  def predict(self, inputs):
    # hidden layer
    hidden_inputs = self.weights_ih.dot(inputs) + self.bias_hidden
    hidden_outputs = self.activation_function("sigmoid", hidden_inputs)

    # output layer
    output_inputs = self.weights_ho.dot(hidden_outputs) + self.bias_output
    outputs = self.activation_function("sigmoid", output_inputs)

    return outputs

  def train(self, training_inputs, training_outputs, batch_size, max_iter = 1000, learning_rate = 0.1):
    for i in range(max_iter):
      # intialize derivative
      grad_weights_ih = np.zeros(self.weights_ih.shape)
      grad_weights_ho = np.zeros(self.weights_ho.shape)
      grad_bias_hidden = np.zeros(self.bias_hidden.shape)
      grad_bias_output = np.zeros(self.bias_output.shape)

      # mini batch
      index = random.sample(range(0, len(training_inputs)), batch_size)
      inputs = np.array([training_inputs[j] for j in index])
      targets = np.array([training_outputs[j] for j in index])

      for j in range(batch_size):
        input = np.array(inputs[j], ndmin = 2).transpose()
        target = np.array(targets[j], ndmin = 2).transpose()

        # FEED FOWARD
        hidden_outputs = self.activation_function("sigmoid", self.weights_ih.dot(input) + self.bias_hidden)
        outputs = self.activation_function("sigmoid", self.weights_ho.dot(hidden_outputs) + self.bias_output)

        # BACK PROPAGATION
        output_errors = target - outputs
        grad_weights_ho += output_errors * outputs * (outputs - 1).dot(hidden_outputs.transpose())
        grad_bias_output += output_errors * outputs * (outputs - 1)

        hidden_erros = self.weights_ho.transpose().dot(output_errors)
        grad_weights_ih += hidden_erros * hidden_outputs * (hidden_outputs - 1).dot(input.transpose())
        grad_bias_hidden += hidden_erros * hidden_outputs * (hidden_outputs - 1)
      
      # update weights and biases
      self.weights_ho -= learning_rate * grad_weights_ho / batch_size
      self.bias_output -= learning_rate * grad_bias_output / batch_size
      self.weights_ih -= learning_rate * grad_weights_ih / batch_size
      self.bias_hidden -= learning_rate * grad_bias_hidden / batch_size

