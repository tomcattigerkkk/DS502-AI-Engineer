import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer, num_filter=32, kernel=(3,3), pad=(1, 1),
               stride=(1, 1), if_pool=False):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?

    conv = mx.sym.Convolution(data=input_layer,
                             num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, # weight=w,
                             no_bias=True)
    relu = mx.sym.Activation(data=conv, act_type='relu')

    layer_s = mx.sym.BatchNorm(relu)
    if if_pool:
        layer_s = mx.sym.Pooling(data=layer_s, name='poing', kernel=(2, 2), stride=(2,2), pool_type='max')
    else:
        layer_s = relu

    return layer_s


# Optional
def inception_layer(input_layer, num1_1x1, num2_1x1, num2_3x3,
                    num3_1x1, num3_5x5, num4_1x1, name):
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    # first 1x1 convolution layer

    cov1 = conv_layer(input_layer, num_filter=num1_1x1, kernel=(1,1), if_pool=True)

    # second 1x1 plus 3x3 convolution layer
    cov2 = conv_layer(input_layer, num_filter=num2_1x1, kernel=(1,1), if_pool=False)
    cov2 = conv_layer(cov2, num_filter=num2_3x3, if_pool=True)

    # third 1x1 plus 5x5 convolution layer
    cov3 = conv_layer(input_layer, num_filter=num3_1x1, kernel=(1, 1), if_pool=False)
    cov3 = conv_layer(cov3, num_filter=num3_5x5, kernel=(5, 5), pad=(2, 2), if_pool=True)

    # forth 3x3 max-pooling plus 1x1 convolutin layer
    cov4 = mx.sym.Pooling(data=input_layer, name='poing', kernel=(3, 3), stride=(1, 1), pool_type='max')
    cov4 = conv_layer(cov4, num_filter=num4_1x1, kernel=(1, 1), stride=(2, 2), if_pool=False)
    # concat
    inception_output = mx.sym.Concat(*[cov1, cov2, cov3, cov4], name=('concat %s' % name))

    return inception_output


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?

    data_f = data

    # create 3 CNN network
    cnn_0 = conv_layer(data_f, num_filter=32, kernel=(3, 3), if_pool=True)
    cnn_1 = conv_layer(cnn_0, num_filter=64, kernel=(3, 3), if_pool=True)
    cnn_2 = conv_layer(cnn_1, num_filter=128, kernel=(3, 3), if_pool=True)

    # add inception layer

    inception_l = inception_layer(cnn_2, 32, 32, 32, 16, 32, 32, 'inception1')

    # flatten CNN
    flatten = mx.symbol.Flatten(data=inception_l)

    # flatten without inception
    # flatten = mx.symbol.Flatten(data=cnn_2)
    # 1st FCN
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=128)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")

    # 2nd FCN
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)

    # softmax output
    cnn_n = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')


    return cnn_n
