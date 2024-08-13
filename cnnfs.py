import numpy as np
import cv2 as cv
import pickle
import copy
import os 



# For CNNs 
class Convolutional_Layer: 

    # Create padded matrix
    @staticmethod
    def pad(matrix, l, r, t, b, padding):
        w = len(matrix[0])
        h = len(matrix)
        o_w = w + r + l 

        if (padding == 'zero'): 
            padding_l = np.array([0.] * l)
            padding_r = np.array([0.] * r)
            padding_t = np.array([[0.] * o_w] * t)
            padding_b = np.array([[0.] * o_w] * b)

            padded_hor = []
            for row in matrix: 
                padded_hor.append(np.concatenate((padding_l, row, padding_r)))

            matrix = np.array(padded_hor)

            if (t == 0 and b > 0): 
                matrix = np.concatenate((matrix, padding_b))
            elif (b == 0 and t > 0): 
                matrix = np.concatenate((padding_t, matrix))
            elif (t > 0 and b > 0):
                matrix = np.concatenate((padding_t, matrix, padding_b))
            else: 
                matrix = matrix

            return matrix
        else: 
            return matrix


    # Create unpadded matrix
    @staticmethod
    def unpad(mat, l, r, t, b, padding):
        if (padding == 'zero'): 
            
            if (b > 0): 
                mat = mat[:-b]
            if (t > 0): 
                mat = mat[t:]

            unpadded = []
            for row in mat: 
                if (r > 0): 
                    row = row[:-r]
                if (l > 0): 
                    row = row[l:]
                unpadded.append(row)

            return np.array(unpadded)
        else: 
            return mat 


    # Correlate matrix and kernel and return the sum
    @staticmethod
    def correlate(mat, ker):
        width = len(mat[0]) - len(ker[0]) + 1
        height = len(mat) - len(ker) + 1
    
        correlated_input = []
    
        for i in range(height): 
            for j in range(width): 
                sum = 0
                for k in range(len(ker)): 
                    for h in range(len(ker[0])): 
                        sum += mat[i + k][j + h] * ker[k][h]
                correlated_input.append(sum)
    
        return np.array(correlated_input).reshape((height, width))

    @staticmethod
    def batch_norm(input, epsilon): 
        sample_size = len(input)
        channel_size = len(input[0])
        h = len(input[0][0])
        w = len(input[0][0][0])
        mean = np.zeros((channel_size, h, w))
        min = input[0]
        max = input[0]
        variance = np.zeros((channel_size, h, w))
        
        for j in range(channel_size):
            for i in range(sample_size):
                mean[j] += input[i][j]

        mean = mean / sample_size

        for i in range(sample_size): 
            variance += ((input[i] - mean)*(input[i] - mean))
            min = np.minimum(min, input[i])
            max = np.maximum(max, input[i])

        variance = variance / sample_size

        norm = []

        for mat in input: 
            norm.append((mat-mean) / np.sqrt(variance + epsilon))

        return np.array(norm)



    @staticmethod
    def rotate(mat): 
        w = len(mat[0])
        h = len(mat)
        mat = mat.reshape(w*h)
        mat = mat[::-1]
        return mat.reshape((h,w))


    #Conctructor: 
    def __init__(self, 
                 input_shape,           # Shape of the input image [num_channels, width, height]  
                 kernel_shape,          # Shape of kernel [k_width, k_height]
                 kernel_per_matrix=1,   # Number of kernel per matrix (number of feature kernels), 
                 stride=1,              # Stride 
                 padding='valid',       # Padding, 'zero-pad' for zero-padding 
                 weight_regularizer_l1=0, 
                 weight_regularizer_l2=0, 
                 bias_regularizer_l1=0, 
                 bias_regularizer_l2=0):   

        # Assign variables
        self.input_shape = input_shape
        self.num_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.kernel_height = kernel_shape[0]
        self.kernel_width = kernel_shape[1]
        self.kernel_per_matrix = kernel_per_matrix
        self.stride = stride
        self.padding = padding
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
        # Valid padding (no padding)
        if (padding == 'valid'): 
            
            self.output_height = ((self.input_height - self.kernel_height) // self.stride) + 1
            self.output_width = ((self.input_width- self.kernel_width) // self.stride) + 1

            # No padding 
            self.padding_w_r = 0
            self.padding_w_l = 0
            self.padding_h_t = 0
            self.padding_h_b = 0
        
        else:
            
            # Width padding 
            padding_w = (self.stride - 1)* self.input_width - self.stride + self.kernel_width
            self.padding_w_l = (padding_w // 2) + (padding_w % 2)
            self.padding_w_r = (padding_w // 2)
            # Height padding
            padding_h = (self.stride - 1)* self.input_height - self.stride + self.kernel_height
            self.padding_h_t = (padding_h // 2) + (padding_h % 2)
            self.padding_h_b = (padding_h // 2)

            self.output_height = self.input_height
            self.output_width = self.input_width

            # Update new input width and height
            self.input_width += padding_w
            self.input_height += padding_h
                
        # Layer's bias
        self.biases = np.zeros((self.kernel_per_matrix, self.output_height, self.output_width))
        # Layer's kernels
        self.kernels = 0.1 * np.random.randn(self.kernel_per_matrix, self.num_channels, self.kernel_width, self.kernel_height)

        # Layer's bias gradient
        self.dbiases = np.zeros(self.biases.shape)
        # Layer's kernel gradient
        self.dkernels = np.zeros(self.kernels.shape)


    # Forward pass
    def forward(self, inputs, training):

        self.inputs = inputs
        self.batch_size = inputs.shape[0]

        self.padded_inputs = inputs

        self.output = np.zeros((self.batch_size, self.kernel_per_matrix, self.output_height, self.output_width))
        
        # Valid padding (no padding)
        if (self.padding != 'valid'): 
             # Padded input
            padded_input = []
            for i in range(self.batch_size):           # Go through each sample in batch
                padded_sample = []
                for j in range(self.num_channels):    # Go through each channel in a sample
                    padded_mat = self.pad(self.inputs[i][j], self.padding_w_l, self.padding_w_r, self.padding_h_t, self.padding_h_b, self.padding)
                    padded_sample.append(padded_mat)
                padded_input.append(padded_sample)
                
            self.padded_inputs = np.array(padded_input)

        for sample_idx in range(self.batch_size):                       # Go through each sample in batch
            for kernel_idx in range(self.kernel_per_matrix):                 # Go through each kernel in kernel set
                for channel_idx in range(self.num_channels):      # Go through each channel
                    self.output[sample_idx][kernel_idx] += self.correlate(self.padded_inputs[sample_idx][channel_idx], self.kernels[kernel_idx][channel_idx])
        
                self.output[sample_idx][kernel_idx] += self.biases[kernel_idx]

        

       
    # Backward pass
    def backward(self, doutput):
#         doutput_pad_v = len(self.kernels[0][0]) - 1
#         doutput_pad_h = len(self.kernels[0][0][0]) - 1
#         
#         for sample_idx in range(self.batch_size):
#             for kernel_idx in range(self.kernel_per_matrix): 
#                 for channel_idx in range(self.num_channels):
#                     kernel_grad = self.correlate(self.padded_inputs[sample_idx][channel_idx], doutput[sample_idx][kernel_idx])
#                     self.dkernels[sample_idx][kernel_idx][channel_idx] = self.correlate(self.padded_inputs[sample_idx][channel_idx], doutput[sample_idx][kernel_idx])
#                 self.dbiases[sample_idx][kernel_idx] = doutput[sample_idx][kernel_idx]
#         
#         self.padded_dinputs = np.zeros(self.padded_inputs.shape)
# 
#         for h in range(self.batch_size): 
#             for i in range(self.num_channels): 
#                 for j in range(self.kernel_per_matrix):
#                     padded_doutput = self.pad(doutput[h][j], doutput_pad_h, doutput_pad_h, doutput_pad_v, doutput_pad_v, 'zero')
#                     rotated_kernel = self.rotate(self.kernels[h][j][i])
# 
#                     self.padded_dinputs[h][i] += self.correlate(padded_doutput, rotated_kernel)
#         
#         trimmed_dinputs = []
#         for i in range(len(self.padded_dinputs)):
#             trimmed_dinput = []
#             for j in range(len(self.padded_dinputs[0])): 
#                 trimmed_dinput.append(self.unpad(self.padded_dinputs[i][j], self.padding_w_l, self.padding_w_r, self.padding_h_t, self.padding_h_b, self.padding))
#             
#             trimmed_dinputs.append(trimmed_dinput)
# 
#         self.dinputs = np.array(trimmed_dinputs)

        self.padded_dinputs = np.zeros(self.padded_inputs.shape)
        self.dkernels = np.zeros(self.dkernels.shape)
        self.dbiases = np.zeros(self.dbiases.shape)
        self.dinputs = np.zeros(self.inputs.shape)
        
        for kernel_idx in range(self.kernel_per_matrix): 
            for sample_idx in range(self.batch_size):
                self.dbiases[kernel_idx] += doutput[sample_idx][kernel_idx]
                for channel_idx in range(self.num_channels):
                    ker_grad = self.correlate(self.inputs[sample_idx][channel_idx], doutput[sample_idx][kernel_idx])
                    self.dkernels[kernel_idx][channel_idx] += ker_grad

        self.dkernels = self.dkernels / self.batch_size
        self.dbiases = self.dbiases / self.batch_size

        doutput_pad_w = self.kernel_width - 1
        doutput_pad_h = self.kernel_height - 1

        for channel_idx in range(self.num_channels):
            for kernel_idx in range(self.kernel_per_matrix):
                for sample_idx in range(self.batch_size):
                    padded_doutput = self.pad(doutput[sample_idx][kernel_idx], doutput_pad_h, doutput_pad_h, doutput_pad_w, doutput_pad_w, 'zero')
                    rotated_kernel = self.rotate(self.kernels[kernel_idx][channel_idx])
                    correlated_mat = self.correlate(padded_doutput, rotated_kernel)
                    self.padded_dinputs[sample_idx][channel_idx] += correlated_mat
                    self.dinputs[sample_idx][channel_idx] += self.unpad(correlated_mat, self.padding_w_l, self.padding_w_r, self.padding_h_t, self.padding_h_b, 'zero')

        self.padded_dinputs = self.padded_dinputs / self.batch_size
        self.dinputs = self.dinputs / self.batch_size

        
        
    def get_parameters(self):
        return self.kernels, self.biases

    def set_parameters(self, kernels, biases): 
        self.kernels = kernels
        self.biases = biases


# Flatten Layer
class Flatten_Layer: 
    def __init__(self): 
        pass

    def forward(self, inputs, training):
        self.inputs = inputs
        self.batch_size = self.inputs.shape[0]
        self.output = []

        for sample in inputs: 
            flattened_input = np.array(sample.flatten())
            self.output.append(flattened_input)

        self.output = np.array(self.output)


    def backward(self, doutput): 
        self.dinputs = doutput.reshape(self.inputs.shape)            
        


class Pool_Layer:
    @staticmethod
    def max_pooling(mat, pool_w, pool_h): 
        mat_w = len(mat[0])
        mat_h = len(mat)
        out_w = mat_w // pool_w
        out_h = mat_h // pool_h
        mul_mat = np.zeros(mat.shape)

        flattened_output = []
        for i in range(0, mat_h, pool_h): 
            for j in range(0, mat_w, pool_w): 
                pool_max = mat[i][j]
                x_ = j
                y_ = i 
                for k in range(pool_h): 
                    for h in range(pool_w):
                        if (pool_max < mat[i+k][j+h]): 
                            pool_max = mat[i+k][j+h]
                            x_ = j + h 
                            y_ = i + k
                mul_mat[y_][x_] = 1
                flattened_output.append(pool_max)


        return np.array(flattened_output).reshape((out_h, out_w)), mul_mat


    def __init__(self, input_shape, pool_type, pool_size):
        self.pool_type = pool_type
        self.pool_w = pool_size[1]
        self.pool_h = pool_size[0]
        self.num_input = input_shape[0]
        self.output_w = input_shape[2] // self.pool_w
        self.output_h = input_shape[1] // self.pool_h

    def forward(self, inputs, training):
        self.inputs = inputs
        self.batch_size = self.inputs.shape[0]
        self.mul_dinputs = np.zeros(self.inputs.shape)
        self.output = np.zeros((self.batch_size, self.num_input, self.output_w, self.output_h))

        for sample_idx in range(self.batch_size): 
            for mat_idx in range(self.num_input): 
                if (self.pool_type == 'max'): 
                    self.output[sample_idx][mat_idx], self.mul_dinputs[sample_idx][mat_idx] = \
                        self.max_pooling(inputs[sample_idx][mat_idx], self.pool_w, self.pool_h)


    def backward(self, doutput):
        
        self.dinputs = np.zeros((self.batch_size, self.num_input, self.output_w * self.pool_w, self.output_h * self.pool_h))

        for l in range(self.batch_size):
            for i in range(self.num_input):
                for y in range(len(doutput[l][i])): 
                    for x in range(len(doutput[l][i][0])):
                        for k in range(self.pool_h): 
                            for h in range(self.pool_w): 
                                y_ = y * self.pool_h + k
                                x_ = x * self.pool_w + h
                                self.dinputs[l][i][y_][x_] = self.mul_dinputs[l][i][y_][x_] * doutput[l][i][y][x]
        
"""
def load_mnist_dataset (dataset, path): 
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels: 
        for file in os.listdir(os.path.join(path, dataset,label)): 
            image = cv.imread(os.path.join(path, dataset, label, file), cv.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    X = np.array(X)[:128]
    y = np.array(y).astype('uint8')[:1000]

    return X,y



def create_data_mnist (path): 
    X,y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test





input = [
    [
        [
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
        [
            [2,3,4,1],
            [7,8,6,5],
            [9,12,10,11],
            [16,13,14,15]
        ],
        [
            [2,1,2,1],
            [7,8,8,1],
            [9,1,1,11],
            [4,3,1,15]
        ]
    ], 
    [
        [
            [1,0,3,1],
            [5,5,5,11],
            [9,6,11,8],
            [0,6,15,5]
        ],
        [
            [0,1,0,1],
            [3,8,2,5],
            [9,4,10,2],
            [9,13,7,9]
        ],
        [
            [4,4,4,2],
            [0,1,2,3],
            [5,5,8,0],
            [2,1,0,4]
        ]
    ]
]


doutput = [
    [
        [
            [1,0,0,0],
            [0,1,0,1],
            [0,0,0,1],
            [1,0,0,0]
        ],
        [
            [0,0,0,0],
            [1,0,0,1],
            [0,1,1,0],
            [0,0,0,1]
        ],
    ], 
    [
        [
            [0,0,0,0],
            [1,0,0,1],
            [0,1,1,0],
            [0,0,0,1]
        ],
        [
            [1,0,0,0],
            [0,1,0,1],
            [0,0,0,1],
            [1,0,0,0]
        ]
    ]
]

pool_output = [
    [
        [
            [0,1],
            [1,0],
        ],
        [
            [1,0],
            [0,1],
        ],
        [
            [2,2],
            [1,1]
        ]
    ], 
    [
        [
            [1,1],
            [1,0],
        ],
        [
            [0,0],
            [1,1],
        ],
        [
            [1,1],
            [0,1]
        ]
    ], 
]

test_input = [
    [
        [
            [0,1,2],
            [3,4,5],
            [6,7,8],
        ],
        [
            [9,10,11],
            [12,13,14],
            [15,16,17],
        ],
    ], 
    [
        [
            [8,7,6],
            [5,4,3],
            [2,1,0],
        ],
        [
            [11,16,9],
            [14,13,12],
            [17,10,17],
        ],
    ], 
]


test_kernels = [
    [
        [
            [0,1],
            [1,0],
        ],
        [
            [1,0],
            [0,1],
        ],
    ], 
    [
        [
            [1,1],
            [1,0],
        ],
        [
            [0,0],
            [1,1],
        ],
    ], 
]


test_biases = [
        [
            [1,2],
            [3,4],
        ],
        [
            [4,3],
            [2,1],
        ],
    ]

test_dinputs = [
    [
        [
            [1,2],
            [0,1],
        ],
        [
            [-2,3],
            [4,0],
        ],
    ], 
    [
        [
            [-1,0],
            [0,1],
        ],
        [
            [0,0],
            [1,-1],
        ],
    ], 
]

test_flattened_output = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])


test_input = np.array(test_input)
test_biases = np.array(test_biases)
test_kernels = np.array(test_kernels)
test_dinputs = np.array(test_dinputs)


input = np.array(input)
doutput = np.array(doutput)

print("Input: \n", input)
print("Input Shape: ", input.shape)
conv1 = Convolutional_Layer(input[0].shape, (3,3), kernel_per_matrix=2, stride=1, padding='zero')

print("Kernels: \n", conv1.kernels)
print("Kernels Shape: ", conv1.kernels.shape)

print("Biases: \n", conv1.biases)
print("Biases Shape: ", conv1.biases.shape)

print("\nPadding: left: ", conv1.padding_w_l, " right: ", conv1.padding_w_r, " top: ", conv1.padding_h_t, " bottom: ", conv1.padding_h_b)

conv1.forward(input, False)

print("Padded Input: \n", conv1.padded_inputs)
print("Padded Input Shape: ", conv1.padded_inputs.shape)

print("Output: \n", conv1.output)
print("Output Shape: ", conv1.output.shape)


conv.backward(doutput)

print("DBiases: \n", conv.dbiases)
print("DBiases Shape: ", conv.dbiases.shape)

print("DKernels: \n", conv.dkernels)
print("DKernels Shape: ", conv.dkernels.shape)

print("Padded dinputs : \n", conv.padded_dinputs)
print("Padded dinputs Shape: ", conv.padded_dinputs.shape)

print("DInputs: \n", conv.dinputs)
print("Dinputs Shape: ", conv.dinputs.shape)

pool_layer = Pool_Layer(conv.output.shape, 'max', (2,2))
pool_layer.forward(conv.output, False)

print("Max pooling output: \n", pool_layer.output)
print("Max pooling output Shape: ", pool_layer.output.shape)

flatten = Flatten_Layer()
flatten.forward(pool_layer.output, False)

print("Flattened output: \n", flatten.output)
print("Flattened output Shape: ", flatten.output.shape)



conv2 = Convolutional_Layer(test_input[0].shape, (2,2), kernel_per_matrix=2,padding='valid')
conv2.set_parameters(test_kernels, test_biases)
conv2.forward(test_input, False)

conv2.backward(test_dinputs)

print("Test kernel gradients: \n", conv2.dkernels)
print("Test bias gradients: \n", conv2.dbiases)
print("Test padded input gradients: \n", conv2.padded_dinputs)
print("Test input gradients: \n", conv2.dinputs)

pool = Pool_Layer(input[0].shape, 'max', (2,2))
pool.forward(input, False)
pool.backward(pool_output)

print("Pool Layer: \n")
print("Pool input: \n", pool.inputs)
print("Pool output: \n", pool.output)
print("Pool grad helper: \n", pool.mul_dinputs)
print("Pool dinput: \n", pool.dinputs)

flatten = Flatten_Layer()
flatten.forward(test_input, False)
flatten.backward(test_flattened_output)

print("Flatten Layer")
print("Flatten output: \n", flatten.output)
print("Flatten dinput: \n", flatten.dinputs)


print("Forward output: \n", conv2.output)



X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')


keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

X = (X - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

X = np.array([X[i].reshape((1,28,28)) for i in range(len(X))])
X_test = np.array([X_test[i].reshape((1,28,28)) for i in range(len(X_test))])

print("\nX shape: ", X.shape)

conv3 = Convolutional_Layer(X[0].shape, (3,3), kernel_per_matrix=20,padding='zero')
pool3 = Pool_Layer((20,28,28), 'max', (4,4))
flatten3 = Flatten_Layer()

conv3.forward(X, True)
pool3.forward(conv3.output, True)
flatten3.forward(pool3.output, True)

print("Flatten Output: ", flatten3.output.shape)


"""

