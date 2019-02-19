import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:

  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batch=False):
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
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.use_batch = use_batch
    self.params['W1'] = weight_scale * np.random.rand(
        num_filters, input_dim[0], filter_size, filter_size)
    #self.params['b1'] = np.zeros(num_filters)

    W2d = (num_filters * int((input_dim[1]-filter_size)/2+1) * int((input_dim[2]-filter_size)/2+1), hidden_dim)
    n2,d2 = W2d
    #print(n2)
    #print(d2)
    self.params['W2'] = weight_scale * np.random.rand(int(n2),int(d2))
    #self.params['W2'] = weight_scale * np.random.rand()
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.rand(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    #batch1 = self.bn_params[1]
    #print(X.shape)
    #conv_out_w = (input_dim[2] - filter_size + 1)
    #conv_out_h = (input_dim[1] - filter_size + 1)
    #conv_out_c = num_filters
    #max_pool_out_w = int(conv_out_w / 2)
    #max_pool_out_h = int(conv_out_h / 2)
    #max_pool_out_c = conv_out_c

    #fc_in_size = max_pool_out_c * max_pool_out_h * max_pool_out_w
    conv_out, conv_cache = conv_forward(X, self.params['W1'])
    mp_out, mp_cache = max_pool_forward(conv_out, pool_param)
    relu1_out, relu1_cache = relu_forward(mp_out)
    relu1_out = relu1_out.reshape(-1,32*11*11)
    fc1_out, fc1_cache = fc_forward(relu1_out,self.params['W2'], self.params['b2'])
    relu2_out, relu2_cache = relu_forward(fc1_out)
    fc2_out, fc2_cache = fc_forward(relu2_out,self.params['W3'], self.params['b3'])

    scores = fc2_out
    #gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    #batch2 = self.bn_params[2]
    #ar_out, ar_cache = affine_batchnorm_relu_forward(crp_out, W2, b2,
    #                                               gamma2, beta2, bn_param2)

    #scores, scores_cache = affine_forward(ar_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    dfc2, dW3, db3 = fc_backward(dscores, fc2_cache)
    dW3 += self.reg * W3

    if self.use_batch:
		pass
	else:
      drelu2 = relu_backward(dfc2, relu2_cache)
      dfc1, dW2, db2 = fc_backward(drelu2, fc1_cache)
      dfc1 = dfc1.reshape(-1,32,11,11)
      drelu1 = relu_backward(dfc1, relu1_cache)
      dmp = max_pool_backward(drelu1, mp_cache)
      dconv, dW1 = conv_backward(dmp, conv_cache)

    dW2 += self.reg * W2

    dW1 += self.reg * W1

    grads['W1'] = dW1
    #grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3

    #grads['gamma1'] = dg1
    #grads['beta1'] = dbeta1
    #grads['gamma2'] = dg2
    #grads['beta2'] = dbeta2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
