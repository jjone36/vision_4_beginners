
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

def init_filter(f, mi, mo, stride):
    '''
    initialize filters
    if input_shape = (a, a, mi), fm_sizes = (f, f, mi), stride = s,
    output_shape = ( (a-f)/s +1, (a-f)/s +1, mo ) where mo = number of filters
    '''
    num = np.random.randn(f, f, mi, mo) * np.sqrt(2./(f*f*mi))
    return num.astype(np.float32)

# Convolutional layer class
class ConvLayer:

    def __init__(self, f, mi, mo, stride = 2, padding = 'VALID'):
        self.W = tf.Variable(init_filter(f, mi, mo, stride))
        self.b = tf.Variable(np.zeros(mo, dtype = np.float32))
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        X = tf.nn.conv2d(input = X,
                        filter = self.W,
                        strides = [1, self.stride, self.stride, 1],
                        padding = self.padding)
        X += self.b
        return X

    # This is for sanity check later
    def copy_keras_layers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))

    def get_params(self):
        return self.W, self.b


# BatchNormalization layer class
class BNLayer:

    def __init__(self, D):
        self.running_mean = tf.Variable(np.zeros(D, dtype = np.float32), trainable = True)
        self.running_var = tf.Variable(np.zeros(D, dtype = np.float32), trainable = True)
        self.beta = tf.Variable(np.zeors(D, dtype = np.float32))
        self.gamma = tf.Variable(np.zeros(D, dtype = np.float32))

    def forward(self, X):
        X = tf.nn.batch_normalization(X,
                                      mean = self.running_mean,
                                      variance = self.running_var,
                                      offset = self.beta,
                                      scale = self.gamma,
                                      variance_epsilon = 1e-3)
        return X

    # This is for sanity check later
    def copy_keras_layers(self, layer):
        gamma, beta, running_mean, running_var = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1, op2, op3, op4))

    def get_params(self):
        return [self.running_mean, self.running_var, self.beta, self.gamma]


# Create ConvBlock class
class ConvBlock:
    # Main branch + Skip connection
    def __init__(self, mi, fm_sizes, stride = 2, activation = tf.nn.relu):

        assert len(fm_sizes) == 3

        self.session = None
        self.activate = activation

        # Main Branch
        # conv -> bn / f = 1, s = 2 --> output_shape = ( (mi - f)/s + 1, (mi - f)/s + 1, mo )
        self.conv_1 = ConvLayer(f = 1, mi = mi, mo = fm_sizes[0], stride)
        self.bn_1 = BNLayer(fm_sizes[0])
        # conv -> bn / f = 3, padding = 'SAME' --> output_shape = ( mi, mi, mo )
        self.conv_2 = ConvLayer(f = 3, mi = fm_sizes[0], mo = fm_sizes[1], stride = 1, padding = 'SAME')
        self.bn_2 = BNLayer(fm_sizes[1])
        # conv -> bn / f = 3 --> output_shape = ( (mi - f)/2 + 1, (mi - f)/2 + 1, mo )
        self.conv_3 = ConvLayer(f = 1, mi = fm_sizes[1], mo = fm_sizes[2], stride = 1)
        self.bn_3 = BNLayer(fm_sizes[2])

        # Skip connection
        # Conv -> bn / f = 1, s = 2 --> output_shape = ( (mi -f)/2 + 1, (mi -f)/2 + 1, mo )
        self.conv_s = ConvLayer(1, mi, fm_sizes[2], stride)
        self.bn_s = BNLayer(fm_sizes[2])

        self.layers = [self.conv_1, self.bn_1,
                       self.conv_2, self.bn_2,
                       self.conv_3, self.bn_3,
                       self.conv_s, self.bn_s]

        self.input = tf.placeholder(tf.float32, shape = (1, 224, 224, mi))
        self.output = self.forward(self.input)

    def feed_forward(self, X):
        # Main brance
        X_1 = self.conv1.forward(X)
        X_1 = self.bn_1.forward(X_1)
        X_1 = self.activate(X_1)

        X_1 = self.conv_2.forward(X_1)
        X_1 = self.bn_2.forward(X_1)
        X_1 = self.activate(X_1)

        X_1 = self.conv_3.forward(X_1)
        X_1 = self.bn_3.forward(X_1)
        X_1 = self.activate(X_1)

        # Shortcut connection
        X_2 = self.conv_s.forward(X)
        X_2 = self.bn_s.forward(X_2)

        # Adding skip connection
        X = X_1 + X_2
        X = self.activate(X)
        return X

    def predict(self, X):
        assert self.session != None
        prediction = self.session.run(self.output, feed_dict = {self.input : X})
        return prediction

    def set_session(self, session):
        self.session = session
        self.conv_1.session = session
        self.bn_1.session = session
        self.conv_2.session = session
        self.bn_2.session = session
        self.conv_3.session = session
        self.bn_3.session = session
        self.conv_s.session = session
        self.bn_s.session = session

    # This is for sanity check later
    def copy_keras_layers(self, layers):
        self.conv_1.copy_keras_layers(layers[0])
        self.bn_1.copy_keras_layers(layers[1])
        self.conv_2.copy_keras_layers(layers[3])
        self.bn_2.copy_keras_layers(layers[4])
        self.conv_3.copy_keras_layers(layers[6])
        self.bn_3.copy_keras_layers(layers[8])
        self.conv_s.copy_keras_layers(layers[7])
        self.bn_s.copy_keras_layers(layers[9])

    def get_params(self):
        params = []
        for layer in self.layers:
            param = layer.get_params()
            params.append(param)
        return params


if __name__=='__name__':
    # Initialize ConvBlock object
    conv_block = ConvBlock(mi = 3, fm_sizes= [64, 64, 256], stride = 1)

    # Fake input image
    X = nprandom.random((1, 224, 224, 3))

    # Run the session
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        conv_block.set_session(session)
        session.run(init)

        prediction = conv_block.predict(X)
        print("prediction shape: ", prediction.shape)
