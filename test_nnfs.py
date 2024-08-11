import pytest
from cnnfs import *
import cv2 as cv
import numpy as np


@pytest.fixture
def data():
    colored_img = cv.imread("/Users/macos/Projects/neuralnetworkfromscratch/cifar-10-images/batch_1/0/213.png", cv.IMREAD_COLOR)
    blue_channel, green_channel, red_channel = cv.split(colored_img)
    channels = np.array([blue_channel, green_channel, red_channel])
    return channels

@pytest.fixture
def convo_lay1(data):
    return Convolutional_Layer(data.shape, (3,3), kernel_per_matrix=2, stride=1, padding='valid')

def test_padding1(convo_lay1):
    assert convo_lay1.padding_w_r == 0
    assert convo_lay1.padding_w_l == 0
    assert convo_lay1.padding_h_t == 0
    assert convo_lay1.padding_h_b == 0
    assert convo_lay1.output.shape == (2, 30, 30)
    assert convo_lay1.input_width == 32
    assert convo_lay1.input_height == 32
    assert convo_lay1.biases.shape == convo_lay1.output.shape
    assert convo_lay1.kernels.shape == (2, 3, 3, 3)
    assert convo_lay1.input_shape == (3, 32, 32)


@pytest.fixture
def convo_lay2(data):
    return Convolutional_Layer(data.shape, (3,3), kernel_per_matrix=2, stride=1, padding='zero')

def test_padding2(convo_lay2):
    assert convo_lay2.padding_w_r == 1
    assert convo_lay2.padding_w_l == 1
    assert convo_lay2.padding_h_t == 1
    assert convo_lay2.padding_h_b == 1 
    assert convo_lay2.output.shape == (2, 32, 32)
    assert convo_lay2.input_width == 34
    assert convo_lay2.input_height == 34
    assert convo_lay2.biases.shape == convo_lay2.output.shape
    assert convo_lay2.kernels.shape == (2, 3, 3, 3)

@pytest.fixture
def convo_lay3(data):
    return Convolutional_Layer(data.shape, (4,4), kernel_per_matrix=4, stride=1, padding='zero')

def test_padding3(convo_lay3):
    assert convo_lay3.padding_w_r == 1
    assert convo_lay3.padding_w_l == 2
    assert convo_lay3.padding_h_t == 2
    assert convo_lay3.padding_h_b == 1
    assert convo_lay3.output.shape == (4,32,32)
    assert convo_lay3.input_width == 35
    assert convo_lay3.input_height == 35
    assert convo_lay3.biases.shape == convo_lay3.output.shape
    assert convo_lay3.kernels.shape == (4, 3, 4, 4)

@pytest.fixture
def convo_lay4(data):
    return Convolutional_Layer((1,5,5), (4,4), kernel_per_matrix=2, stride=2, padding='zero')

def test_padding4(convo_lay4):
    assert convo_lay4.padding_w_r == 3
    assert convo_lay4.padding_w_l == 4
    assert convo_lay4.padding_h_t == 4
    assert convo_lay4.padding_h_b == 3
    assert convo_lay4.output.shape == (2,5,5)
    assert convo_lay4.input_width == 12
    assert convo_lay4.input_height == 12
    assert convo_lay4.biases.shape == convo_lay4.output.shape
    assert convo_lay4.kernels.shape == (2, 1, 4, 4)

@pytest.fixture
def data_1x2x2():
    return np.array([[  [1,2],
                        [3,4]   ]])

@pytest.fixture
def convo_lay5(data_1x2x2):
    return Convolutional_Layer(data_1x2x2.shape, (2,2), kernel_per_matrix=1, stride=1, padding='valid')

def test_forward_backward_1x2x2(convo_lay5, data_1x2x2):
    assert convo_lay5.input_shape == (1,2,2)
    convo_lay5.biases = np.array([[1]])
    convo_lay5.kernels = np.array([[[[1,0], [0,1]]]])
    convo_lay5.forward(data_1x2x2, False)
    assert convo_lay5.output == np.array([[[6]]])
    doutput = np.array([[[-1]]])
    convo_lay5.backward(doutput)
    assert np.all(convo_lay5.dkernels) == np.all(np.array([[[[-1. -2.], [-3. -4.]]]]))
    assert np.all(convo_lay5.dbiases) == np.all(doutput)
    assert np.all(convo_lay5.dinputs) == np.all(np.array([[[-1,0],[0, -1]]]))


@pytest.fixture
def data_1x3x3():
    return np.array([[  [1,2,3],
                        [3,4,5],[6,7,8]   ]])

@pytest.fixture
def convo_lay6(data_1x3x3):
    return Convolutional_Layer(data_1x3x3.shape, (3,3), kernel_per_matrix=1, stride=1, padding='zero')

def test_forward_backward_1x3x3(convo_lay6, data_1x3x3):
    assert convo_lay6.input_shape == (1,3,3)
    convo_lay6.biases = np.array([[[1,2,3], [4,5,6], [7,8,9]]])
    convo_lay6.kernels = np.array([[[[1,0,0], [1,0,1], [0,1,0]]]])
    convo_lay6.forward(data_1x3x3, False)
    print(convo_lay6.output)


@pytest.fixture
def data_4x4(): 

    return np.array([   [   [1,2,3,5],
                            [6,7,9,10],
                            [0,1,1,2],
                            [4,2,5,2] ], 
                        
                        [   [0,1,2,3], 
                            [4,2,5,2], 
                            [1,4,3,4],
                            [22,4,34,2] ]   ])


def test_flatten_4x4(data_4x4):
    flatten_layer = Flatten_Layer()
    flatten_layer.forward(data_4x4[0],False)
    assert np.all(flatten_layer.output) == np.all(np.array([[1,2,3,5,6,7,9,10,0,1,1,2,4,2,5,2]]))


def test_pool_4x4_2(data_4x4):
    assert np.all(Pool_Layer.max_pooling(data_4x4[0], 2,2)[0]) == np.all(np.array([[7,10],[4,5]]))
    pool_layer = Pool_Layer(data_4x4.shape, 'max', (2,2))
    pool_layer.forward(data_4x4, False)
    assert np.all(pool_layer.output) == np.all(np.array([[[ 7., 10.], [ 4., 5.]], [[ 4., 5.], [22.,34.]]]))
    doutput = np.array([[[ 1 , 2], [ 3, 4]], [[ 5, 6], [7, 8]]])
    print(pool_layer.backward(doutput))

    

