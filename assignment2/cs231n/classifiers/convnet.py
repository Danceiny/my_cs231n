import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    First Edition:
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    Second Edition:
    A arbitrary number of hidden layers convolutional network with the following
    architecture:

    [conv-relu-pool]xN - [conv - relu] - [affine]xM - [softmax or SVM]

    [conv-relu-pool]XN - [affine]XM - [softmax or SVM]

    ' - [conv - relu] - ' can be trun on/off by parameter 'connect_conv'
    both architecture can do with or without batch normalization

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    The learnable parameters of the model are stored in the dictionary self.params
    that maps parameter names to numpy arrays.
    """


    def __init__(self, conv_dims, hidden_dims, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False, loss_function="softmax",connect_conv=0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        # self.use_batchnorm = use_batchnorm
        # self.params = {}
        # self.reg = reg
        # self.dtype = dtype

        self.num_conv_layers = len(conv_dims)
        self.num_fc_layers = len(hidden_dims)
        self.use_connect_conv = connect_conv > 0
        self.use_batchnorm = use_batchnorm
        self.loss_function = loss_function
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################

    #     # Size of the input
    #     C, H, W = input_dim

    #     # Conv layer
    #     # The parameters of the conv is of size (F,C,HH,WW) with
    #     # F give the nb of filters, C,HH,WW characterize the size of
    #     # each filter
    #     # Input size : (N,C,H,W)
    #     # Output size : (N,F,Hc,Wc)
    #     F = num_filters
    #     filter_height = filter_size
    #     filter_width = filter_size
    #     stride_conv = 1  # stride
    #     P = (filter_size - 1) / 2  # padd
    #     Hc = (H + 2 * P - filter_height) / stride_conv + 1
    #     Wc = (W + 2 * P - filter_width) / stride_conv + 1

    #     W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
    #     b1 = np.zeros(F)

    #     # Pool layer : 2*2
    #     # The pool layer has no parameters but is important in the
    #     # count of dimension.
    #     # Input : (N,F,Hc,Wc)
    #     # Ouput : (N,F,Hp,Wp)

    #     width_pool = 2
    #     height_pool = 2
    #     stride_pool = 2
    #     Hp = (Hc - height_pool) / stride_pool + 1
    #     Wp = (Wc - width_pool) / stride_pool + 1

    #     # Hidden Affine layer
    #     # Size of the parameter (F*Hp*Wp,H1)
    #     # Input: (N,F*Hp*Wp)
    #     # Output: (N,Hh)

    #     Hh = hidden_dim
    #     W2 = weight_scale * np.random.randn(F * Hp * Wp, Hh)
    #     b2 = np.zeros(Hh)

    #     # Output affine layer
    #     # Size of the parameter (Hh,C)
    #     # Input: (N,Hh)
    #     # Output: (N,C)

    #     C = num_classes
    #     W3 = weight_scale * np.random.randn(Hh, C)
    #     b3 = np.zeros(C)

    #     self.params.update({'W1': W1,
    #                         'W2': W2,
    #                         'W3': W3,
    #                         'b1': b1,
    #                         'b2': b2,
    #                         'b3': b3})

    #     for k, v in self.params.iteritems():
    #         self.params[k] = v.astype(dtype)

    # def loss(self, X, y=None):
    #     """
    #     Evaluate loss and gradient for the three-layer convolutional network.

    #     Input / output: Same API as TwoLayerNet in fc_net.py.
    #     """

    #     N = X.shape[0]

    #     W1, b1 = self.params['W1'], self.params['b1']
    #     W2, b2 = self.params['W2'], self.params['b2']
    #     W3, b3 = self.params['W3'], self.params['b3']

    #     # pass conv_param to the forward pass for the convolutional layer
    #     filter_size = W1.shape[2]
    #     conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    #     # pass pool_param to the forward pass for the max-pooling layer
    #     pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    #     scores = None
    #     #######################################################################
    #     # TODO: Implement the forward pass for the three-layer convolutional net,  #
    #     # computing the class scores for X and storing them in the scores          #
    #     # variable.                                                                #
    #     #######################################################################

    #     # Forward into the conv layer
    #     x = X
    #     w = W1
    #     b = b1
    #     conv_layer, cache_conv_layer = conv_relu_pool_forward(
    #         x, w, b, conv_param, pool_param)
    #     N, F, Hp, Wp = conv_layer.shape  # output shape

    #     # Forward into the hidden layer
    #     x = conv_layer.reshape((N, F * Hp * Wp))
    #     w = W2
    #     b = b2
        
    #     hidden_layer, cache_hidden_layer = affine_relu_forward(x, w, b)
    #     N, Hh = hidden_layer.shape

    #     # Forward into the linear output layer
    #     x = hidden_layer
    #     w = W3
    #     b = b3
    #     scores, cache_scores = affine_forward(x, w, b)

    #     if y is None:
    #         return scores

    #     loss, grads = 0, {}
    #     #######################################################################
    #     # TODO: Implement the backward pass for the three-layer convolutional net, #
    #     # storing the loss and gradients in the loss and grads variables. Compute  #
    #     # data loss using softmax, and make sure that grads[k] holds the gradients #
    #     # for self.params[k]. Don't forget to add L2 regularization!               #
    #     #######################################################################

    #     data_loss, dscores = softmax_loss(scores, y)
    #     reg_loss = 0.5 * self.reg * np.sum(W1**2)
    #     reg_loss += 0.5 * self.reg * np.sum(W2**2)
    #     reg_loss += 0.5 * self.reg * np.sum(W3**2)
    #     loss = data_loss + reg_loss

    #     # Backpropagation
    #     grads = {}
    #     # Backprop into output layer
    #     dx3, dW3, db3 = affine_backward(dscores, cache_scores)
    #     dW3 += self.reg * W3

    #     # Backprop into first layer
    #     dx2, dW2, db2 = affine_relu_backward(dx3, cache_hidden_layer)
    #     dW2 += self.reg * W2

    #     # Backprop into the conv layer
    #     dx2 = dx2.reshape(N, F, Hp, Wp)
    #     dx, dW1, db1 = conv_relu_pool_backward(dx2, cache_conv_layer)
    #     dW1 += self.reg * W1

    #     grads.update({'W1': dW1,
    #                   'b1': db1,
    #                   'W2': dW2,
    #                   'b2': db2,
    #                   'W3': dW3,
    #                   'b3': db3})

    #     #######################################################################
    #     #                             END OF YOUR CODE                             #
    #     #######################################################################

    #     return loss, grads


        # initialize conv_layers:
        for i in xrange(self.num_conv_layers):
            if i == 0:
                self.params['CW1'] = weight_scale * np.random.randn(\
                                                                    conv_dims[i][0], \
                                                                    input_dim[0], \
                                                                    conv_dims[i][1], \
                                                                    conv_dims[i][1])
                self.params['cb1'] = np.zeros(conv_dims[i][0])
                # print self.params['CW1'].shape,self.params['cb1'].shape
                if use_batchnorm:
                    self.params['cgamma1'] = np.ones(conv_dims[i][0])
                    self.params['cbeta1'] = np.zeros(conv_dims[i][0])

            else:
                self.params['CW' + str(i + 1)] = weight_scale * np.random.randn(\
                                                                    conv_dims[i][0], \
                                                                    conv_dims[i - 1][0], \
                                                                    conv_dims[i][1], \
                                                                    conv_dims[i][1])
                self.params['cb' + str(i + 1)] = np.zeros(conv_dims[i][0])
                if use_batchnorm:
                    self.params['cgamma' + str(i + 1)] = np.ones(conv_dims[i][0])
                    self.params['cbeta' + str(i + 1)] = np.zeros(conv_dims[i][0])

        if self.use_connect_conv:
            self.params['CCW'] = weight_scale * np.random.randn(\
                                                                    connect_conv[0], \
                                                                    conv_dims[-1][0], \
                                                                    connect_conv[1], \
                                                                    connect_conv[1])
            self.params['ccb'] = np.zeros(connect_conv[0])
            if self.use_batchnorm:
                self.params['ccgamma'] = np.ones(connect_conv[0])
                self.params['ccbeta'] = np.zeros(connect_conv[0])

        # initialize affine layers:
        for i in xrange(self.num_fc_layers):
                if i == 0:
                    # initialize first affine layers
                    if self.use_connect_conv:
                        self.params['FW' + str(i + 1)] = weight_scale * np.random.randn(
                             connect_conv[0] * input_dim[1] * input_dim[2] / 4 ** self.num_conv_layers,
                             hidden_dims[i])
                    else:
                        self.params['FW' + str(i + 1)] = weight_scale * np.random.randn(
                             conv_dims[-1][0] * input_dim[1] * input_dim[2] / 4 ** self.num_conv_layers,
                             hidden_dims[i])
                    self.params['fb' + str(i + 1)] = np.zeros(hidden_dims[i])
                    # initialize first batch normalize layers
                    if self.use_batchnorm:
                        self.params['fgamma' + str(i + 1)] = np.ones(hidden_dims[i])
                        self.params['fbeta' + str(i + 1)] = np.zeros(hidden_dims[i])
                elif i == self.num_fc_layers - 1:
                    # initialize last affine layers
                    self.params['FW' + str(i + 1)] = \
                            weight_scale * np.random.randn(hidden_dims[i - 1], num_classes)
                    self.params['fb' + str(i + 1)] = np.zeros(num_classes)
                    if self.use_batchnorm:
                        self.params['fgamma' + str(i + 1)] = np.ones(num_classes)
                        self.params['fbeta' + str(i + 1)] = np.zeros(num_classes)
                else:
                    # initialize    affine layers
                    self.params['FW' + str(i + 1)] = \
                            weight_scale * np.random.randn(hidden_dims[i - 1], hidden_dims[i])
                    self.params['fb' + str(i + 1)] = np.zeros(hidden_dims[i])
                    # initialize batch normalize layers
                    if self.use_batchnorm:
                        self.params['fgamma' + str(i + 1)] = np.ones(hidden_dims[i])
                        self.params['fbeta' + str(i + 1)] = np.zeros(hidden_dims[i])

        # pass conv_params to the forward pass for the convolutional layer
        self.conv_params = []
        self.conv_params = [{'stride': 1, 'pad': (conv_dims[i][1] - 1) / 2}\
                                                for i in xrange(self.num_conv_layers)]
        if self.use_connect_conv:
            self.conv_params.append({'stride': 1, 'pad': (connect_conv[1] - 1) / 2})
        # pass pool_param to the forward pass for the max-pooling layer
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.sbn_params = []
        if self.use_batchnorm:
            self.sbn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers)]
            if self.use_connect_conv:
                self.sbn_params.append({'mode': 'train'})
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_fc_layers)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
            for bn_param in self.sbn_params:
                bn_param['mode'] = mode
                # bn_param[mode] = mode
                # the original CODE is right above, BUT I think it's wrong
            if self.use_connect_conv:
                self.sbn_params[-1]['mode'] = mode

        scores = None
        ############################################################################
        # Implement the forward pass for the three-layer convolutional net,                #
        # computing the class scores for X and storing them in the scores                    #
        # variable.                                                                                                                                #
        ############################################################################
        a = X
        cache = []
        # conv_forward:
        for i in xrange(self.num_conv_layers):
            if self.use_batchnorm:
                a_temp, cache_temp = conv_bn_relu_pool_forward(a, \
                                                                                self.params['CW' + str(i + 1)], \
                                                                                self.params['cb' + str(i + 1)], \
                                                                                self.params['cgamma' + str(i + 1)], \
                                                                                self.params['cbeta' + str(i + 1)], \
                                                                                self.conv_params[i], \
                                                                                self.pool_param, \
                                                                                self.sbn_params[i])
                a = a_temp
                cache.append(cache_temp)
            else:
                a_temp, cache_temp = conv_relu_pool_forward(a, \
                                                                                self.params['CW' + str(i + 1)], \
                                                                                self.params['cb' + str(i + 1)], \
                                                                                self.conv_params[i], \
                                                                                self.pool_param)
                a = a_temp
                cache.append(cache_temp)

        # connect_conv_forward
        if self.use_connect_conv:
            if self.use_batchnorm:
                # conv_forward:
                a_temp, cache_temp = conv_forward_fast(a, \
                                                                                self.params['CCW'], \
                                                                                self.params['ccb'], \
                                                                                self.conv_params[-1])
                a = a_temp
                cache.append(cache_temp)

                # spatial_batchnorm_forward:
                a_temp, cache_temp = spatial_batchnorm_forward(a, \
                                                                                self.params['ccgamma'], \
                                                                                self.params['ccbeta'], \
                                                                                self.sbn_params[-1])
                a = a_temp
                cache.append(cache_temp)

                # Rulu forward:
                a_temp, cache_temp = relu_forward(a)
                a = a_temp
                cache.append(cache_temp)
            else:
                a_temp, cache_temp = conv_relu_forward(a, \
                                                                                self.params['CCW'], \
                                                                                self.params['ccb'], \
                                                                                self.conv_params[-1])
                a = a_temp
                cache.append(cache_temp)

        # affine forward:
        N = X.shape[0]
        x_temp_shape = a.shape
        for i in xrange(self.num_fc_layers):
            a_temp, cache_temp = affine_forward(a.reshape(N, -1), \
                                                                                    self.params['FW' + str(i + 1)], \
                                                                                    self.params['fb' + str(i + 1)])
            a = a_temp
            cache.append(cache_temp)
            if self.use_batchnorm:
                a_temp, cache_temp = batchnorm_forward(a, \
                                                                                    self.params['fgamma' + str(i + 1)], \
                                                                                    self.params['fbeta' + str(i + 1)],
                                                                                    self.bn_params[i])
                a = a_temp
                cache.append(cache_temp)
        scores = a
        ############################################################################
        #                                                         END OF YOUR CODE                                                         #
        ############################################################################

        if mode == 'test':
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,             #
        # storing the loss and gradients in the loss and grads variables. Compute    #
        # data loss using softmax or svm_loss, and make sure that grads[k] holds     #
        # the gradients for self.params[k]. Don't forget to add L2 regularization! #
        ############################################################################
        # compute loss:
        if self.loss_function == 'softmax':
            loss_without_reg, dscores = softmax_loss(scores, y)
        elif self.loss_function == 'svm_loss':
            loss_without_reg, dscores = svm_loss(scores, y)
        loss = loss_without_reg
        for i in xrange(self.num_conv_layers):
            loss += 0.5 * self.reg * np.sum(self.params['CW' + str(i + 1)] ** 2)
        if self.use_connect_conv:
            loss += 0.5 * self.reg * np.sum(self.params['CCW'] ** 2)
        for i in xrange(self.num_fc_layers):
            loss += 0.5 * self.reg * np.sum(self.params['FW' + str(i + 1)] ** 2)

        #### compute fully-connected layers grads{}
        dout = dscores
        for i in reversed(xrange(self.num_fc_layers)):
            if self.use_batchnorm:
                dout_temp, grads['fgamma' + str(i + 1)], grads['fbeta' + str(i + 1)] = \
                                            batchnorm_backward(dout, cache.pop(-1))
                dout = dout_temp
            dout_temp, grads['FW' + str(i + 1)], grads['fb' + str(i + 1)] = \
                                            affine_backward(dout, cache.pop(-1))
            dout = dout_temp

        # compute connect conv_layer grads{}
        dout = dout.reshape(x_temp_shape)
        if self.use_connect_conv:
            if self.use_batchnorm:
                dout_temp = relu_backward(dout, cache.pop(-1))
                dout = dout_temp
                dout_temp, grads['ccgamma'], grads['ccbeta'] = \
                        spatial_batchnorm_backward(dout, cache.pop(-1))
                dout = dout_temp
                dout_temp, grads['CCW'], grads['ccb'] = \
                    conv_backward_fast(dout, cache.pop(-1))
                dout = dout_temp
            else:
                dout_temp, grads['CCW'], grads['ccb'] = \
                        conv_relu_backward(dout, cache.pop(-1))
                dout = dout_temp

        # compute conv_layers grads{}
        for i in reversed(xrange(self.num_conv_layers)):
            if self.use_batchnorm:
                dout_temp, grads['CW' + str(i + 1)], grads['cb' + str(i + 1)], \
                    grads['cgamma' + str(i + 1)], grads['cbeta' + str(i + 1)] = \
                                            conv_bn_relu_pool_backward(dout, cache.pop(-1))
                dout = dout_temp
            else:
                dout_temp, grads['CW' + str(i + 1)], grads['cb' + str(i + 1)] = \
                                            conv_relu_pool_backward(dout, cache.pop(-1))
                dout = dout_temp
        ############################################################################
        #                                                         END OF YOUR CODE                                                         #
        ############################################################################

        return loss, grads
