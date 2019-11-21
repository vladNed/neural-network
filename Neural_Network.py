import matplotlib.pyplot as plt
import numpy as np
import pickle

class NeuralNetwork(object):

    def __init__(self,no_features,no_neurons,learning_rate=2e-3):
        self.weights1 = np.random.rand(no_features,no_neurons)
        self.no_neurons = no_neurons
        self.weights2 = np.random.rand(no_neurons,1)
        self.learning_rate = learning_rate
        #print('Initialized Network with weights : \nWeights 1: \n%s \nWeights 2: \n%s \n\n' % (self.weights1,self.weights2))
    
    def activation_layer_one(self,Z):
        return np.tanh(Z)
    
    def activation_layer_one_prime(self,z):
        return 1 - np.power(np.tanh(z), 2)
    
    def sigmoid(self,Z):
        return 1 / (1+np.exp(-Z))   

    def sigmoid_der(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x)) 

    def train_network(self,X_train,Y_train,epochs):

        loss_list = []
        for _ in range(epochs):
            errors = []

            for sample,label in zip(X_train,Y_train):

                # Hidden Layer 1
                summation_1 = np.dot(sample,self.weights1)
                function_1 = self.activation_layer_one(summation_1)

                # Output Layer
                summation_2 = np.dot(function_1,self.weights2)
                prediction = self.sigmoid(summation_2)
                
                # Calculate error
                error = prediction - label
                errors.append(error) 

                # Backpropagation step 1
                delta2 = self.sigmoid_der(prediction)
                z_delta2 = error * delta2

                delta1 = self.activation_layer_one_prime(function_1)
                z_delta1 = error * delta1
    
                # Backpropagation step 2
                self.weights2 = (self.weights2.T - (self.learning_rate * function_1.T * z_delta2)).T
                for i in range(len(self.weights1)):
                    self.weights1[i] = (self.weights1[i].T - (self.learning_rate * sample[i] * z_delta1))
            
            loss_list.append(sum(np.power(errors,2)))
            #ceva = ((1 / 2) * (np.power((errors), 2)))
            #loss_list.append(sum(ceva))

        print('Loss: %s' % loss_list[-1])
        plt.plot(loss_list)
        plt.show()
    
    def save_model(self):
        with open('nn_weights.pickle','wb') as handle:
            pickle.dump({'weights1': self.weights1, 'weights2': self.weights2},handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_weights(self,path=None):
        with open(path,'rb') as handle:
            weights = pickle.load(handle)
        
        self.weights1 = weights['weights1']
        self.weights2 = weights['weights2']

    def predict(self,X_predict):
        # Hidden Layer 1
        summation_1 = np.dot(X_predict,self.weights1)
        function_1 = self.activation_layer_one(summation_1)

        # Output Layer
        summation_2 = np.dot(function_1,self.weights2)
        prediction = self.sigmoid(summation_2)
        return int(round(prediction[0]))