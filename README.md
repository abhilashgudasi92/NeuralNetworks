# Neural Networks

Pair programming assignment CS 6375: Machine Learning, Spring 2018 <br>

**Team members:**
- Ankita Patil
- Abhilash Gudasi

<hr>

- In this program, a neural net having two hidden layers is created. But note that, you can change the number of nodes in the hidden layer
- Use of activation functions
  - sigmoid
  - tanh
  - Relu
  
- The neural net is tested on following three datasets:
  - **Car Evaluation Dataset**: 
    https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
  - **Iris Dataset**: 
    https://archive.ics.uci.edu/ml/datasets/Iris
  - **Adult Census Income Dataset**: 
    https://archive.ics.uci.edu/ml/datasets/Census+Income
- We have tried to tune the following parameters and presented our analysis in the report. Also detailed logs can be found <a href="https://github.com/patilankita79/NeuralNetworks/tree/master/Results">here.</a>
  - learning rate
  - number of iterations
  - number of nodes in the hidden layer
  
<hr>

## Instructions to run the code

- **Preprocessing datasets:**<br>
 Each folder inside NeuralNet/ i.e Adult,Car and Iris has preProcessDataset.py python file which needs to be run.<br>
 This preprocesses data and then creates .csv file in the current directory. After which train and test data .csv file is created based on splitting argument user  entered
 
 **Syntax:**
 
 ```
  python preProcessDataset.py <url for dataset> <train_test_split>
 ```

 **Example:**
 
 ```
   python preProcessDataset.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data 0.2
	 python preProcessDataset.py https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data 0.2
	 python preProcessDataset.py https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data 0.2
 ```

In our project folder, we have already generated preProcessedDatataset for respective datasets and train.csv, test.csv from the processedDataset [You can skip running preProcessDataset.py]

- Each folder inside NeuralNet/ i.e Adult,Car and Iris has a NeuralNet.py python file which needs to be run.

- From the command line, go to the respective folder of the .py file running the algorithm to get neural network, training it and then finding error running the network  built on test dataset. 


 **Syntax:**
 
 ```
 python NeuralNet.py <activation_function> <learning_rate> <#iterations> <# Nodes in hidden layer 1> <# Nodes in hidden layer 2>
 ```

 **Example:**
 
 ```
  python NeuralNet.py sigmoid 0.1 1000 4 2
 ```
 
- We have considered different cases in order to evaluate the model. Hence, according change the parameters
- Example, **changing the learning rate**

```
  python NeuralNet.py sigmoid 0.01 1000 4 2
```

**Changing the number of nodes in hidden layer**

```
  python NeuralNet.py sigmoid 0.05 1000 5 3
```

**Changing the number of iterations**

```
python NeuralNet.py relu 0.01 2000 4 2
```


