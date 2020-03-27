
import numpy as np


x=np.array(([1,1,1]), dtype=float)
print(x.shape)

rand_weights_1 = np.random.rand(3,4)
rand_weights_2 = np.random.rand(4, 1)
print(x.shape)
print(x)
print(rand_weights_1)
print(rand_weights_2)

results = np.dot(x, rand_weights_1)

print("layer 1 neuron values")
print(results)
print("adsf")

# activation function
def sigmoid(t):
    return 1 / (1.+np.exp(-t))


layer_1 = sigmoid(results)
print(layer_1)

layer_2 = sigmoid(np.dot(layer_1, rand_weights_2))
print("output")
print(layer_2)

