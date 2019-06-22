import numpy as np
import tensorflow as tf

from ResNet_convblock import ConvLayer, BNLayer


class IdentityBlock:
    '''
    create identity block object which has an identity mapping (No ConvLayer and BN)
    '''
    def __init__(self, mi, fm_sizes, stride = 2, activation):

        assert len(fm_sizes == 3)

        self.session = None
        self.activate = activation

        # Main branch
        # ConvBLock_1 : conv -> bn / f = 1, s = 2 --> output_shape = ( (mi - f)/s + 1, (mi - f)/s + 1, mo )
        self.conv_1 = ConvLayer(f = 1, mi = mi, mo = fm_sizes[0], stride)
        self.bn_1 = BNLayer(fm_sizes[0])
        # ConvBLock_2 : conv -> bn / f = 3, padding = 'SAME' --> output_shape = ( mi, mi, mo )
        self.conv_2 = ConvLayer(f = 3, fm_sizes[0], fm_sizes[1], stride = 1, padding = 'SAME')
        self.bn_2 = BNLayer(fm_sizes[1])
        # ConvBLock_3 : conv -> bn / f = 3 --> output_shape = ( (mi - f)/2 + 1, (mi - f)/2 + 1, mo )
        self.con_3 = ConvLayer(f = 1, fm_sizes[1], fm_sizes[2], stride = 1)
        self.bn_3 = BNLayer(fm_sizes[2])

        self.layers = [self.conv_1, self.bn_1,
                       self.conv_2, self.bn_2,
                       self.conv_3, self.bn_3]

        # In case when input is not passed from the start
        self.input = tf.placeholder(tf.float32, shape = (1, 224, 224, mi))
        self.output = self.forward(self.input)

    def feed_forward(self, X):
        # Main Branch
        X_1 = self.conv_1.forward(X)
        X_1 = self.bn_1.forward(X_1)
        X_1 = self.activate(X_1)

        X_1 = self.conv_2.forward(X_1)
        X_1 = self.bn_2.forward(X_1)
        X_1 = self.activate(X_1)

        X_1 = self.conv_3.forward(X_1)
        X_1 = self.bn_3.forward(X_1)

        # Adding skip connection
        X += X_1
        X = self.activate(X)
        return X


    def predict(self, X):
        assert self.session != None
        prediction = self.session.sun(self.output, feed_dict = {self.input : X})
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
        assert len(layers) == 10
        self.conv_1.copy_keras_layers(layers[0])
        self.bn_1.copy_keras_layers(layers[1])
        self.conv_2.copy_keras_layers(layers[3])
        self.bn_2.copy_keras_layers(layers[4])
        self.conv_3.copy_keras_layers(layers[6])
        self.bn_3.copy_keras_layers(layers[7])


    def get_params(self):
        params = []
        for layer in self.layers:
            param = layer.get_params()
            params.append(param)
        return params


if __name__=='__main__':
    # Initialize IdentityBlock object
    iden_block = IdentityBlock()

    # Fake input image
    X = np.random.random((1, 224, 224, 256))

    # Run the session
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        conv_block.set_session(session)
        session.run(init)

        prediction = iden_block.predict(X)
        print("prediction shape: ", prediction.shape)
