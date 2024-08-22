import numpy as np 
import matplotlib.pyplot as plt
import os
from cnnfs import * 
import cv2 as cv
import pickle
import copy

def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def sine_data(samples=1000):

    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)

    return X, y



# Layer Dense 
class Layer_Dense:

    #Contructor
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # random weights 
        self.biases = np.zeros((1, n_neurons))  
        
        # Regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training): 
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues): # pass gradient to backward pass
        # Gradient of weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        # L2 weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient of values
        self.dinputs = np.dot(dvalues, self.weights.T)


    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases): 
        self.weights = weights
        self.biases = biases


class Layer_Dropout: 

    #Constructor
    def __init__(self, q): 
        self.rate = 1 - q

    # Forward pass
    def forward(self, inputs, training): 
        self.inputs = inputs
        
        if not training: 
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    # Backward
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask



# Input layer
class Layer_Input: 
    #Forward
    def forward(self, inputs,training):
        self.output = inputs
        

# ReLU Activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values
        self.output = np.maximum(0, inputs) # if inputs[i] >= 0; output[i] = inputs[i] else output[i] = 0;

    # Backward pass
    def backward(self, dvalues):
        # Copy inputs gradient to modify
        self.dinputs = dvalues.copy()
 
        # Zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0

    

    def predictions(self, outputs):
        return outputs


# Softmax Activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs, training): 
        # Pass inputs to instance
        self.inputs = inputs
        
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    
    # Backward pass
    def backward(self, dvalues): 
        
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Emurate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1,1)
            # Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) # diagflat is for the diagonal matrix with dvalues's shape

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Sigmoid Activation
class Activation_Sigmoid: 
    
    # Forward 
    def forward(self, inputs, training ): 
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward
    def backward(self, dvalues): 
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (output > 0.5) * 1

# Linear Activation
class Activation_Linear: 

    # Forward
    def forward(self, inputs, training): 
        self.inputs = inputs
        self.output = inputs

    # Backward
    def backward(self, dvalues): 
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - lr = 1.0 for now 
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Initialize params
    def pre_update_params(self):
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        if self.momentum: 
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else: 
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases


        layer.weights += weight_updates
        layer.biases += bias_updates

    # After params are updated
    def post_update_params(self): 
        self.iterations += 1


class Optimizer_Adagrad: 

    # Intialize optimizer
    def __init__(self, learning_rate=1., decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Initialize params
    def pre_update_params(self):
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    # Update params
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'): 
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Update and normalize SGD params with squared rooted cache
        layer.weights += -self.current_learning_late * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_late * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # After update
    def post_update_params(self):
        self.iterations += 1


# RMSprop optimizer
class Optimizer_RMSprop: 

    # Initialize optimizer 
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Initialize params
    def pre_update_params(self):
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + (self.decay * self.iterations)))


    # Update params
    def update_params(self, layer):
        
        if not hasattr(layer, 'weight_cache'): 
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

        # Update and normalize SGD params with squared rooted cache
        layer.weights += -self.current_learning_late * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_late * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # After update
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam: 

    # Initialize optimizer 
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Initialize params
    def pre_update_params(self):
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + (self.decay * self.iterations)))


    # Update params
    def update_params(self, layer):
        
        if hasattr(layer, 'weights'): 
            weights_kernels = layer.weights
        elif hasattr(layer, 'kernels'): 
            weights_kernels = layer.kernels
        
        if not hasattr(layer, 'weight_cache'):    
            layer.weight_momentums = np.zeros_like(weights_kernels)
            layer.weight_cache = np.zeros_like(weights_kernels)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        if hasattr(layer, 'dweights'): 
            d_weights_kernels = layer.dweights
        elif hasattr(layer, 'dkernels'): 
            d_weights_kernels = layer.dkernels
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * d_weights_kernels
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1**(self.iterations + 1))

        # Cache or velocity term
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * d_weights_kernels ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases ** 2

        # Corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        #Vanilla SGD param updated + RMS
        weights_kernels += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # After update
    def post_update_params(self):
        self.iterations += 1


# Common Loss 
class Loss:

    # Regularization loss calculation
    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers: 

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
    
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
    
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
    
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    # Calculate the data and regularization losses given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False): 
        
        # Calculate sample losses
        sample_losses = self.forward(output,y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses) 
        # self.accumulated_count += len(sample_losses)
        self.accumulated_count += len(y)

        if not include_regularization: 
            return data_loss

        # Return
        return data_loss, self.regularization_loss()

    # Calculate accumulative loss
    def calculate_accumulated(self, *, include_regularization=False): 
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization: 
            return data_loss

        return data_loss, self.regularization_loss()

    def new_pass(self): 
        self.accumulated_sum = 0 
        self.accumulated_count = 0 


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss): 
    
    # Forward pass
    def forward(self, y_pred, y_true): # y_pred: values from neural network, y_true: values from training values
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    # Backward
    def backward(self, dvalues, y_true): 

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true]

        # Calculate gradient 
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples



# Softmax classifier - combined Softmax and CCEL for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        #Calculate gradient 
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient   
        self.dinputs = self.dinputs / samples


# Binary Cross-Entropy Loss function
class Loss_BinaryCrossentropy(Loss): 

    # Forward
    def forward(self, y_pred, y_true): 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    # Backward 
    def backward(self, dvalues, y_true): 
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1-y_true)/(1-clipped_dvalues)) / outputs
        self.dinputs = self.dinputs/samples


# Mean Squared Error Loss - L2
class Loss_MeanSquaredError(Loss): 

    # Forward
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis= -1)
        return sample_losses

    # Backward
    def backward(self, dvalues, y_true): 
        samples = len(dvalues) 
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

# Mean Absolute Error Loss - L1
class Loss_MeanAbsoluteError(Loss):

    # Forward
    def forward(self, y_pred, y_true):
        samples_losses = np.mean(np.abs(y_true-y_pred), axis=-1)
        return sample_losses

    # Backward
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient with all samples
        self.dinputs = self.dinputs / samples 

# Accuracy 
class Accuracy: 
    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y) 
        
        accuracy = np.mean(comparisons)
        
        self.accumulated_sum += np.sum(comparisons) 
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self): 
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy): 

    def __init__ (self): 
        self.precision = None

    def init(self, y, reinit=False): 
        if self.precision is None or reinit: 
            self.precision = np.std(y) / 250

    def compare(self, predictions, y): 
        return np.absolute(predictions - y) < self.precision


class Accuracy_Categorical(Accuracy):

    def init(self,y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y


# Model 
class Model:

    # Constructor
    def __init__(self):
        # List of sequential layers 
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    
    # Add layers to the list
    def add(self, layer): 
        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        if loss is not None:
            self.loss = loss
        if optimizer is not None: 
            self.optimizer = optimizer
        if accuracy is not None: 
            self.accuracy = accuracy

    # Finalize the model and ready for testing 
    def finalize(self): 
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count): 
            # First layer => prev is input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # In-between layers
            elif i < layer_count-1: 
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # Last layer
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # If layer contains an attribute "weights" => it is a trainable layer, add to layer lists
            if hasattr(self.layers[i], 'weights') or hasattr(self.layers[i], 'kernels'):
                self.trainable_layers.append(self.layers[i])

            
            if self.loss is not None: 
                self.loss.remember_trainable_layers(self.trainable_layers)

        if  isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossentropy): 
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()




    # Train model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Init accuracy object
        self.accuracy.init(y)

        train_steps = 1

        if validation_data is not None: 
            validation_steps = 1

            X_val, y_val = validation_data

        if batch_size is not None: 
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None: 
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1


        #Training
        for epoch in range(1, epochs+1):

            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):

                if batch_size is None: 
                    batch_X = X
                    batch_y = y
                else: 
                    batch_X = X[step*batch_size: (step+1)*batch_size]
                    batch_y = y[step*batch_size: (step+1)*batch_size]
                
                output = self.forward(batch_X, training=True)
                
                # print("Output of forward: \n", output)

                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
    
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)

                # print("Predictions: \n", predictions )
                
                accuracy = self.accuracy.calculate(predictions, batch_y)
        
                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers: 
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                #if not step % print_every or step == train_steps - 1: 
                print(  f'step: {step}, ' + 
                            f'acc: {accuracy:.3f}, ' + 
                            f'loss: {loss:.3f}, ' +
                            f'data_loss: {data_loss:.3f}, ' + 
                            f'reg_loss: {regularization_loss:.3f}, ' + 
                            f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(  f'training, ' + 
                f'acc: {epoch_accuracy:.3f}, ' + 
                f'loss: {epoch_loss:.3f}, ' + 
                f'data_loss: {epoch_data_loss:.3f}, ' + 
                f'reg_loss: {epoch_regularization_loss:.3f}, ' + 
                f'lr: {self.optimizer.current_learning_rate}')
            
            if validation_data is not None: 
                self.evaluate(*validation_data, batch_size=batch_size)


    def evaluate(self, X_val, y_val, *, batch_size=None): 

        validation_steps = 1

        if batch_size is not None: 
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps): 

            if batch_size is None: 
                batch_X = X_val
                batch_y = y_val
            else: 
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' + 
            f'acc: {validation_accuracy:.3f}, ' + 
            f'loss: {validation_loss:.3f}')


    def forward(self, X, training):
#         if (len(X.shape) == 4):
#             y_return = []
#             y_full = []
# 
#             for X_ in X: 
#                 self.input_layer.forward(X_, training)
# 
#                 for layer in self.layers: 
#                     layer.forward(layer.prev.output, training)
#                 y_return.append(np.argmax(layer.output))
#                 y_full.append(layer.output)
# 
#             return np.array(y_return), np.array(y_full)
        

        self.input_layer.forward(X, training)
        
        for layer in self.layers: 
            layer.forward(layer.prev.output, training)

        return layer.output  

    def backward(self, output, y):


        if self.softmax_classifier_output is not None: 
            
            self.softmax_classifier_output.backward(output,y)
            
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers): 
            layer.backward(layer.next.dinputs)


    def get_parameters(self): 
        parameters = []
        
        for layer in self.trainable_layers: 
            parameters.append(layer.get_parameters())

        return parameters


    def set_parameters(self, parameters_list):
        for param_item, layer in zip(parameters_list, self.trainable_layers): 
            layer.set_parameters(*param_item)

    def save_parameters(self, path):
        with open(path, 'wb') as f: 
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):

        with open(path, 'rb') as f: 
            self.set_parameters(pickle.load(f))

    def save(self, path): 
        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers: 
            properties = ['inputs', 'output', 'dinputs', 'dweights', 'dbiases', "dkernels"]
            for property in properties: 
                layer.__dict__.pop(property, None)

        with open(path, 'wb')  as f: 
            pickle.dump(model, f)

    def predict(self, X, *, batch_size=None): 
        prediction_steps = 1

        if batch_size is not None: 
            prediction_steps = len(X) // batch_size

            if prediction_steps* batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps): 

            if batch_size is None: 
                batch_X = X 

            else: 
                batch_X = X[step* batch_size: (step+1)*batch_size]

            batch_output = self.forward(batch_X, training=False)

            output.append(batch_output)


        return np.vstack(output)


    @staticmethod
    def load(path):
        with open(path, 'rb') as f: 
            model = pickle.load(f)

            return model



def load_mnist_dataset (dataset, path): 
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels: 
        for file in os.listdir(os.path.join(path, dataset,label)): 
            image = cv.imread(os.path.join(path, dataset, label, file), cv.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')



def create_data_mnist (path): 
    X,y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test



def load_cifar_dataset (path): 
    
    batches = os.listdir(path)

    X = []
    y = []
   
    for batch in batches:

        path_batch = os.path.join(path,batch)

        labels = os.listdir(path_batch)


        for label in labels: 
            for file in os.listdir(os.path.join(path_batch, label)): 
                image = cv.imread(os.path.join(path_batch, label, file), cv.IMREAD_COLOR)

                X.append(image)
                y.append(label)

    X = np.array(X)
    y = np.array(y).astype('uint8')
    
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)

    X = X[keys]
    y = y[keys]

    X = (X - 127.5) / 127.5

    X_ = []

    for x in X: 
        X_.append(np.transpose(x, (2,0,1)))

    X = np.array(X_)
    
    length = X.shape[0] * 9 // 10

    X_test = X[length:]
    y_test = y[length:]
    X = X[:length]
    y = y[:length]
    
    return X, y, X_test, y_test



X, y, X_test, y_test = load_cifar_dataset('cifar-10-images/')


model = Model()


model.add(Convolutional_Layer(X[0].shape, (5,5), kernel_per_matrix=1, stride=1, padding='valid'))
model.add(Activation_ReLU())
model.add(Pool_Layer((1,28,28), 'max', (2,2)))
model.add(Convolutional_Layer((1,14,14), (5,5), kernel_per_matrix=2, stride=1, padding='valid'))
model.add(Activation_ReLU())
model.add(Flatten_Layer())
model.add(Layer_Dense(200, 10))
model.add(Activation_Softmax())

model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-4),
        accuracy=Accuracy_Categorical()
    )



model.finalize()

model.train(X,y,validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

model.save('fashion_mnist_conv.parms')

fashion_mnist_labels = ['T-shirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


"""

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')


keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

X = (X - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

X = np.array([X[i].reshape((1,28,28)) for i in range(len(X))])
X_test = np.array([X_test[i].reshape((1,28,28)) for i in range(len(X_test))])


model = Model()

"""
"""

##### FULLY CONNECTED EXAMPLE
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-4),
        accuracy=Accuracy_Categorical()
    )
#####
"""
"""

image_data = cv.imread("/Users/macos/neuralnetworkfromscratch/fashion_mnist_images/test/1/0012.png", cv.IMREAD_COLOR)

image_data = cv.resize(image_data, (28,28))

image_data = (image_data.reshape(1,-1).astype(np.float32) -127.5) / 127.5

predictions = model_load.predict(image_data)

predictions = model_load.output_layer_activation.predictions(predictions)

prediction = fashion_mnist_labels[predictions[0]]

print(prediction)


# model.load_parameters('fashion_mnist.parms')

# model.evaluate(X_test, y_test)

# model.save_parameters('fashion_mnist.parms')

# model.save('fashion_mnist.model')

# model_load = Model.load('fashion_mnist.model')

fashion_mnist_labels = ['T-shirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# model_load.evaluate(X_test, y_test)

# confidences = model_load.predict(X_test[:5])
# predictions = model_load.output_layer_activation.predictions(confidences)

# print(predictions)

image_data = cv.imread("/Users/macos/neuralnetworkfromscratch/fashion_mnist_images/test/1/0012.png", cv.IMREAD_GRAYSCALE)

image_data = cv.resize(image_data, (28,28))

image_data = (image_data.reshape(1,-1).astype(np.float32) -127.5) / 127.5

predictions = model_load.predict(image_data)

predictions = model_load.output_layer_activation.predictions(predictions)

prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
"""
"""
colored_img = cv.imread("/Users/macos/Projects/neuralnetworkfromscratch/cifar-10-images/batch_1/0/213.png", cv.IMREAD_COLOR)
# print(colored_img)
# cv.imshow('Colored Image', colored_img)

blue_channel, green_channel, red_channel = cv.split(colored_img)
channels = np.array([blue_channel, green_channel, red_channel])
"""
"""
for i in range(32):
    for j in range(32):
        colored_img[i][j][0] = 0
        colored_img[i][j][2] = 0

cv.imshow('Blue Channel', colored_img)
key=cv.waitKey(0)
if key == 27: 
    cv.destroyAllWindows()
else: 
    plt.imshow(colored_img)
    plt.title('Blue Channel')
    plt.show()
"""
"""
print(colored_img.shape)
cv.imshow('Colored Image', colored_img)
key = cv.waitKey(0) 

if key == 27: 
    cv.destroyAllWindows()
else:
    #rgb_img = cv.cvtColor(colored_img, cv.COLOR_BGR2RGB)
    plt.imshow(colored_img)
    plt.title('Colored Image')
    plt.show()
"""

