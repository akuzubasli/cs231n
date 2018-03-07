import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] += -X[i]
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg * W
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y]
  margins = scores - correct_class_scores[:, np.newaxis] + 1
  margins[np.arange(num_train), y] = 0 # we dont want the loss by correct classes (~delta)
  loss = np.sum(np.maximum(0, margins))/num_train
  loss += 0.5 * reg * np.sum(W * W) # regularization
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # SVM max-hinge loss, tüm yanlış classların scorelarının
  # doğru class scorundan delta kadar az olmasını istiyor. Eğer delta kadar azsa ok demektir ve loss'u artırmıyor.
  # gradient update için backpropagation yapmıyor. Yanlış olan her bir class, margin 0'dan büyükse bu miktarda loss'u artırıyor ve
  # yanlış her bir class ilgili weight'in (W*X) azaltılması için o class'a denk gelen X[i] kadar azaltılması için 
  # backpropagation -(dW = X[i] > 0) yapıyor. 
  # diğer yandan, margin > 0 olan yanlış her bir class doğru class'ın scorunun X[i] kadar artırılması için 
  # -(dW = -X[i]) backpropagation yapıyor.
  
  # Bu nedenle, yanlış classlardan margin > 0 olanları belirlememiz ve onlara göre 
  # doğru ve yanlış class weightleri için update vermemiz gerekiyor. 
  # Bunu vectorize edebilmek için margin > 0 maskesi oluşturuyoruz. 
  
  margin_mask = np.zeros(margins.shape)
  margin_mask[margins > 0] = 1 # eğer margin 0'dan büyükse 1. ama doğru classları ayırmamız lazım.
  margin_mask[np.arange(num_train), y] = 0
  margin_mask[np.arange(num_train), y] = - np.sum(margin_mask, axis = 1) # o satırda kaç tane margin > 0 saydık.
  dW = X.T.dot(margin_mask)
  
  # ANCAK! Her sample'dan alınan class weight update kararı, tüm sample sayısına göre average yapılması gerekiyor.
  dW /= num_train # average out the weights stated by each sample x[i]
  dW += reg * W # regularize the weights
  

  return loss, dW
