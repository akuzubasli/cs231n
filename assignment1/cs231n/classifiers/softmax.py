import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, dim = X.shape
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in range(num_train):
    f_i = X[i].dot(W)
    f_i += -np.max(f_i) # to avoid instability
    sum_k = np.sum(np.exp(f_i))
    p = np.exp(f_i) / sum_k
    loss += -np.log(p[y[i]])
    for k in range(num_class):
      dW[:, k] += X[i] * (p[k] - 1 * (k == y[i]))
      
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, dim = X.shape
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f_i = X.dot(W)
  f_i += -np.max(f_i, axis = 1, keepdims = True) # to avoid instability
  sum_k = np.sum(np.exp(f_i), axis = 1, keepdims = True)
  p = np.exp(f_i) / sum_k
  loss += np.sum(-np.log(p[np.arange(num_train), y]))
  p_mask = np.zeros_like(p)
  p_mask[np.arange(num_train), y] = 1
  dW = X.T.dot(p - p_mask)
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

