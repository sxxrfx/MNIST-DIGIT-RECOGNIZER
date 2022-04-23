import numpy as np
from scipy.special import expit
import json

EPOCHS = 30
INPUT_NODES = 28 * 28
HIDDEN_NODES = 16
OUTPUT_NODES = 10
LEARNING_RATE = 0.08
NN_ARCH = (INPUT_NODES, HIDDEN_NODES, HIDDEN_NODES, OUTPUT_NODES)


def sigmoid(z):
    return expit(z)


def dSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


def ReLu(z):
    return np.maximum(0.0, z)


def dRelu(z):
    return np.where(z <= 0, 0.0, 1.0)
    # return 1.0 * (z > 0)


class Neural_Network:

    def __init__(self,
                 input_nodes=INPUT_NODES,
                 hidden_nodes=HIDDEN_NODES,
                 output_nodes=OUTPUT_NODES,
                 learning_rate=LEARNING_RATE,
                 epochs=EPOCHS,
                 nn_arch=NN_ARCH,
                 from_file=False):
        self.num_layers = len(nn_arch)
        # self.num_layers = 4
        self.nn_arch = nn_arch
        self.epochs = epochs
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        print(f'Epochs :: {self.epochs}')
        if not from_file:
            self.biases = [np.random.randn(y, 1) for y in self.nn_arch[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(
                self.nn_arch[:-1], self.nn_arch[1:])]
        else:
            self.__import__()

        self.activation_function = lambda x: sigmoid(x)
        self.d_activation_function = lambda x: dSigmoid(x)
        # self.activation_function = lambda x: ReLu(x)
        # self.d_activation_function = lambda x: dRelu(x)
        self.lr = learning_rate

        pass

    # def train(self, image_list, label_list, epochs=10):
    def train(self, x_train, y_train):
        for e in range(self.epochs):
            print(f"Epoch: {e+1}")
            for i in range(len(x_train)):
                self.__train__(x_train=x_train,
                               y_train=y_train, i=i)
        pass

    def __train__(self, x_train, y_train, i):
        image_list = (x_train[i] / 255.0 * 0.99) + 0.01
        label_list = np.zeros(self.onodes) + 0.01
        label_list[int(y_train[i])] = 0.99
        x = np.array(image_list, ndmin=2).T
        y = np.array(label_list, ndmin=2).T
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(self.lr)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.lr)*nb
                       for b, nb in zip(self.biases, nabla_b)]
        pass

    def feedforward(self, a, i):
        return self.activation_function(np.dot(self.weights[i], a)+self.biases[i])

    def predict(self, inputs_list):
        x = np.array(inputs_list, ndmin=2).T
        for i in range(3):
            x = self.feedforward(x, i)
        return x
        # return softmax(x)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = []
        activations.append(x)
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        delta = self.cost_derivative(
            activations[-1], y) * self.d_activation_function(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.d_activation_function(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def export(self):
        bias_1 = (self.biases[0]).tolist()
        bias_2 = (self.biases[1]).tolist()
        bias_3 = (self.biases[2]).tolist()
        weight_input_to_hl1 = (self.weights[0]).tolist()
        weight_hl1_to_hl2 = (self.weights[1]).tolist()
        weight_hl2_output = (self.weights[2]).tolist()
        weight_dict = {"bias_1": bias_1, "bias_2": bias_2, "bias_3": bias_3,
                       "weight_input_to_hl1": weight_input_to_hl1, "weight_hl1_to_hl2": weight_hl1_to_hl2, "weight_hl2_output": weight_hl2_output}

        json_obj = json.dumps(weight_dict, sort_keys=False, indent=4)

        with open("../model/model.json", "w") as f:
            f.write(json_obj)
        pass

    def test(self, x_test, y_test):
        N = len(x_test)
        n = 0
        for x, y in zip(x_test, y_test):
            ans = self.predict(x)
            if np.argmax(ans) == y:
                n += 1
        print(f"Test samples :: {N}\nAccuracy of the Model :: {n/N*100}%")

    def __import__(self, filepath="model/model.json"):
        with open(filepath, "r") as f:
            weight_dict = json.load(f)

        self.biases = []
        self.weights = []
        self.biases.append(np.array(weight_dict["bias_1"]))
        self.biases.append(np.array(weight_dict["bias_2"]))
        self.biases.append(np.array(weight_dict["bias_3"]))
        self.weights.append(np.array(weight_dict["weight_input_to_hl1"]))
        self.weights.append(np.array(weight_dict["weight_hl1_to_hl2"]))
        self.weights.append(np.array(weight_dict["weight_hl2_output"]))

        pass
