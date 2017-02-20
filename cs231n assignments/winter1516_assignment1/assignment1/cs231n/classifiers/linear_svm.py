import numpy as np
from random import shuffle

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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    diff_count = 0 # initialize additions to gradient at 0
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        diff_count += 1 # if the margin is above 0 increase the gradient
        
        # simply counting the number of classes that didn't meet the desired margin
        # and hence contributed to the loss function
        dW[:, j] += X[i] # gradient update for incorrect rows
        
        loss += margin
    dW[:, y[i]] += -diff_count * X[i] # gradient update for correct row

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # also average the gradient
  dW += reg*W # regularize the weights

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  delta = 1.0
    
#  # Semi vectorized implementation from lecture notes
#  for i in range(num_train):
#        scores = X[i].dot(W)
#        correct_class_score = scores[y[i]]

#        # compute the margins for all classes in one vector op
#        margins = np.maximum(0, scores - correct_class_score + delta)

#        # on y-th position scores[y] - scores[y] canceled and gave delta. We want
#        # to ignore the y-th position and only consider margin on max wrong class
#        margins[y[i]] = 0
#        loss += np.sum(margins)

  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train),y]
    
  # calculate margins across all rows 
  margins = np.maximum(0, scores - correct_class_score[:,np.newaxis] + delta)
    
  # on the y-th position scores[y] - scores[y] canceled so want to ignor y-th
  # position and only consider margin on max wrong class so set equal to zero
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins)

  loss /= num_train # get mean
  loss += 0.5 * reg * np.sum(W*W) # regularization

  # Now implement a vectorized version of the gradient for the structured SVM 
  # loss, and store the result in dW.                                           

  # Vectorized version
  # First create a mask such that the columns map to the classes, and the rows
  # map to the samples

  # Each value in X_mask[i, j] adds a row sample i to column class j multiplied by its value
  # Thus we set equal to 1
    
  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1 # if we clear the margin increase the weights
  
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train),y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train # average gradient weights
  dW += reg * W # add regularization

  return loss, dW
