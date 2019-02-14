import numpy as np

from layers import *

class SVM(object):
  """
  A binary SVM classifier with optional hidden layers.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    self.params['b1'] = np.zeros(1)
    self.hidden = hidden_dim != None
    if hidden_dim != None:
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(1)
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, 1))
    else:
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, 1))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the classification
    score for X[i].

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    layer1, cache1 = fc_forward(X, self.params['W1'], self.params['b1'])
    if self.hidden is True:
        layer2, relucache = relu_forward(layer1)
        scores, cache = fc_forward(layer2, self.params['W2'], self.params['b2'])
    else:
        scores = layer1
        cache  = cache1
    N, _ = scores.shape
    scores = scores.reshape((N,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      #scores[scores < 1] = -1
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    scores = scores.reshape((N, 1))
    y = y*2 - 1
    y1 = y.reshape(y.shape[0],1)
    if self.hidden is True:
        loss, grad2 = svm_loss(scores, y1)
        dx, grads['W2'], grads['b2'] = fc_backward(grad2, cache)
        grads['W2'] = grads['W2'] + self.reg*self.params['W2']
        relugrad = relu_backward(dx, relucache)
        _, grads['W1'], grads['b1'] = fc_backward(relugrad, cache1)
        grads['W1'] = grads['W1'] + self.reg*self.params['W1']
    else:
        loss, grad1 = svm_loss(scores, y1)
        _ ,grads['W1'],grads['b1'] = fc_backward(grad1, cache)
        grads['W1'] = grads['W1'] + self.reg*self.params['W1']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
