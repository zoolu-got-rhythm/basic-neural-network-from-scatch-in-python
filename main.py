import numpy as np


# why do people make X uppercase?
X_train = np.array(([0,0,1], [0,1,1]), dtype=float)
y_train = np.array(([0], [1]), dtype=float)


# activation function
def sigmoid(t):
    return 1 / (1.+np.exp(-t))


# derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# loss function
def cost_function(y_predicted, y_actual):
    # square difference to make a positive (absolute) value
    return (y_actual - y_predicted)**2  # power of 2 = squared


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.layer2_output = np.zeros(self.y.shape) # layer 2 output will be final output/result/prediction

    # should be private?
    def feed_forward(self, x):
        self.layer1_output = sigmoid(np.dot(x, self.weights1))
        self.layer2_output = sigmoid(np.dot(self.layer1_output, self.weights2))
        return self.layer2_output

    # should be private
    def back_propagation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        # deep learning back propagation chain rule calculus vid: https://www.youtube.com/watch?v=tIeHLnjs5U8
        derivative_of_loss_function_wrt_weights2 = \
            np.dot(self.layer1_output.T, (2*(self.y - self.layer2_output) * sigmoid_derivative(self.layer2_output)))

        derivative_of_loss_function_wrt_weights1 = \
            np.dot(self.input.T, (np.dot(2*(self.y - self.layer2_output) * sigmoid_derivative(self.layer2_output),
                                         self.weights2.T) * sigmoid_derivative(self.layer1_output)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += derivative_of_loss_function_wrt_weights1
        self.weights2 += derivative_of_loss_function_wrt_weights2

    # public method: train
    def fit(self, n_of_epochs):
        for i in range(n_of_epochs):
            self.feed_forward(self.input)
            self.back_propagation()

    # public method
    def predict(self, x):
        # if shape of x matrix not equal to shape of input matrix for training throw error
        return self.feed_forward(x)


NN = NeuralNetwork(X_train, y_train)
for i in range(10):
    # mean sum squared loss
    loss = np.mean(np.square(y_train - NN.predict(X_train)))
    print("loss for epoch: " + str(i * 100))
    print(str(loss))
    NN.fit(100)

# could put output layer results into a softmax function instead of sigmoid? to produce whole number labels?
print(NN.predict(np.array([0,0,1])))
print(NN.predict(np.array([0,1,1])))



















