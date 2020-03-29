
import numpy as np


x=np.array(([1,1,1]), dtype=float)
print(x.shape)


rand_weights_1 = np.random.rand(3,4)
rand_weights_2 = np.random.rand(4, 1)
print(x.shape)
print(x)


print("random weights to layer 1")
print(rand_weights_1)


print("random weights to layer 2 (output layer)")
print(rand_weights_2)


results = np.dot(x, rand_weights_1)


print("layer 1 neuron values before applying activation function (sigma)")
print(results)
print("adsf")


# activation function
def sigmoid(t):
    return 1 / (1.+np.exp(-t))


layer_1 = sigmoid(results)
print("layer 1 neurons after applying sigma function to each neuron to squishify - "
      "value between 0 and 1")
print(layer_1)


layer_2 = sigmoid(np.dot(layer_1, rand_weights_2))
print("layer 2 output value: after applying sigma")
print(layer_2)


def cost_function(y_predicted, y_actual):
    # square difference to make a positive (absolute) value
    return (y_actual - y_predicted)**2  # power of 2 = squared






