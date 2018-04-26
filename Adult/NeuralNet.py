#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import sys,os
#import requests
import io
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, train, h1, h2,header = True):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        train_dataset = pd.read_csv(train)                                #self.preprocess(sys.argv[1],float(sys.argv[4]))        #preprocess(url,split)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    # activation function selection
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self,x)
        elif activation == "relu":
            self.__relu(self,x)

    #activation derivative selection
    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self,x)
        elif activation == "relu":
            self.__relu_derivative(self,x)

    # sigmoid activation function
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # tanh activation function
    def __tanh(self, x):
        return np.tanh(x)

    # derivative of tanh function, indicates confidence about existing weight
    def __tanh_derivative(self, x):
        return 1 - np.square(np.tanh(x))

    # relu activation function
    def __relu(self, x):
        return x * (x>0)

    # derivative of relu function, indicates confidence about existing weight
    def __relu_derivative(self, x):
        return 1 * (x > 0)

    # pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc

# =============================================================================
#     def preprocess(self,url,split):
#         #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#         #s = requests.get(url).content
#         names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
#                  "relationship", "race", "sex",
#                  "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
#         # Importing the datasets
#         dataset = pd.read_csv(url, names=names)     #dataset = pd.read_csv(io.StringIO(s.decode('utf-8')), names=names)
# 
#         # mark ? values as missing or NaN
#         dataset.replace(to_replace="[?]", value=np.nan, regex=True, inplace=True)
# 
#         dataset = dataset.dropna()
#         X = dataset.iloc[:, :-1].values
#         y = dataset.iloc[:, 14].values
# 
#         df = pd.DataFrame(dataset)
# 
#         label_encoder = LabelEncoder()
#         df["workclass"] = label_encoder.fit_transform(df["workclass"])
#         df["education"] = label_encoder.fit_transform(df["education"])
#         df["marital-status"] = label_encoder.fit_transform(df["marital-status"])
#         df["occupation"] = label_encoder.fit_transform(df["occupation"])
#         df["relationship"] = label_encoder.fit_transform(df["relationship"])
#         df["race"] = label_encoder.fit_transform(df["race"])
#         df["sex"] = label_encoder.fit_transform(df["sex"])
#         df["native-country"] = label_encoder.fit_transform(df["native-country"])
#         df["class"] = label_encoder.fit_transform(df["class"])
# 
#         # standardizing the dataset
#         df["age"] = preprocessing.scale(df["age"])
#         df["workclass"] = preprocessing.scale(df["workclass"])
#         df["fnlwgt"] = preprocessing.scale(df["fnlwgt"])
#         df["education"] = preprocessing.scale(df["education"])
#         df["education-num"] = preprocessing.scale(df["education-num"])
#         df["marital-status"] = preprocessing.scale(df["marital-status"])
#         df["occupation"] = preprocessing.scale(df["occupation"])
#         df["relationship"] = preprocessing.scale(df["relationship"])
#         df["race"] = preprocessing.scale(df["race"])
#         df["sex"] = preprocessing.scale(df["sex"])
#         df["capital-gain"] = preprocessing.scale(df["capital-gain"])
#         df["capital-loss"] = preprocessing.scale(df["capital-loss"])
#         df["hours-per-week"] = preprocessing.scale(df["hours-per-week"])
#         df["native-country"] = preprocessing.scale(df["native-country"])
# 
#         # writing pre-processed data to a csv file
#         df.to_csv('preProcessedAdultDataset.csv', sep=',', index=False, header=False)
# 
#         # split dataset in to train and test
#         train,test = train_test_split(df, test_size=split)
#         train.to_csv('train.csv', sep=',', index=False, header=False)
#         test.to_csv('test.csv', sep=',', index=False, header=False)
#         return train
# =============================================================================

    # Below is the training function

    def train(self, activation, learning_rate, max_iterations):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation=activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation=activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
        #print("***************" + str(self.w01) + str(self.w12) + str(self.w23))
        print("Activation Layer: " + activation)
        print("The final weight vectors are (starting from input to output layers)")
        print("Layer1 weights:")
        print(self.w01)
        print("Layer2 weights:")
        print(self.w12)
        print("Layer3 weights:")
        print(self.w23)
        print("After " + str(max_iterations) + " iterations, the train error is " + str(np.sum(error) / len(error)))

    def forward_pass(self,activation):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out

    #forward pass for test data
    def forward_pass_test(self,test,activation,h1,h2):
        #h1 = 4
        #h2 = 2
        ncolstest = len(test.columns)
        nrowstest = len(test.index)
        Xtest = test.iloc[:, 0:(ncolstest - 1)].values.reshape(nrowstest, ncolstest - 1)
        ytest = test.iloc[:, (ncolstest - 1)].values.reshape(nrowstest, 1)

        self.X01 = Xtest
        self.X12 = np.zeros((len(Xtest), h1))
        self.X23 = np.zeros((len(Xtest), h2))

        # pass our inputs through our neural network
        in1 = np.dot(Xtest, self.w01 )
        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation == "relu":
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        #error calculation
        error = 0.5 * np.power((out - ytest), 2)
        return error

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output


    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2


    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1


    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
        self.delta01 = delta_input_layer

    # predict function for applying the trained model on the  test dataset.
    # assumed that the test dataset has the same format as the training dataset
    # Output the test error from this function

    def predict(self, test, activation, h1, h2,header = True):
        testdataset = pd.read_csv(test)
        error = self.forward_pass_test(testdataset,activation,h1,h2)
        errPerc = np.sum(error)/len(error)
        #("***************" + str(self.w01) + str(self.w12) + str(self.w23))
        print("Activation Layer: " + activation)
        print("Test dataset error is " + str(errPerc))
        return 0


if __name__ == "__main__":

    iterations = int(sys.argv[3])  # Number of iteration (Example -> 1000)
    hNode1 = int(sys.argv[4]) # Number of nodes in hidden layer 1
    hNode2 = int(sys.argv[5]) # Number of nodes in hidden layer 2
    activation = sys.argv[1]  # Activation functions ("tanh" / "sigmoid" / "relu")
    learning_rate = float(sys.argv[2])  # Learning Rate(Example ->0.05/0.1/0.2)
    neural_network = NeuralNet(os.getcwd() + r"\train.csv",hNode1,hNode2)
    neural_network.train(activation, learning_rate, iterations)
    testError = neural_network.predict(os.getcwd() + r"\test.csv", activation,hNode1, hNode2)
