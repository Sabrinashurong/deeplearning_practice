import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Get shapes
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #print("num_train: ",num_train,"num_classes",num_classes)
  #print("X.shape: ",X.shape,"W.shape: ",W.shape)
  # X.shape = 500, 3073       W.shape = 3073, 10
  

  # iterate over training samples
  # for each training sample, multiply by the weights for each class
  # then f_i.shape will be (10,) and the 10 numbers will represent
  # the weight assigned to each class for a given sample - i.e. our class scores
  for i in range(num_train):
    f_i = X[i].dot(W)
    #print("f_i.shape: ",f_i.shape)
    
    # now normalize to avoid dividing by large numbers later on
    # by shifting the class scores so the highest value is zero
    f_i -= np.max(f_i)

    # compute the loss
    sum_j = np.sum(np.exp(f_i)) # sum of all class scores
    p = lambda x: np.exp(f_i[x]) / sum_j # our softmax nodes
    loss -= np.log(p(y[i]))
   
    # Compute gradient
    for class_ in range(num_classes):
        p_class = p(class_)
        
        # update the weights
        # by subtracting doing prob is/isn't vs. softmax probs
        # and multiply by X-sample (broadcasting)
        # (boolean values automatically convert to 1 or 0)
        dW[:,class_] += (p_class - (class_ == y[i])) * X[i]

        
        # dW.shape = (3073,10)
        # dW[:,class_].shape = (3073,)
        
  loss /= num_train # get average loss
  loss += 0.5 * reg * np.sum(W*W) # apply regularization to loss
  dW /= num_train # average the gradient too
  dW += reg * W # regularize gradient too
  
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  f = X.dot(W) # class scores
    
  # shift to avoid large vals - need keepdims to have an empty dimension to broadcast into
  f -= np.max(f, axis=1, keepdims=True)
  
  # compute the loss
  sum_f = np.sum(np.exp(f),axis=1,keepdims=True) 
  p = np.exp(f) / sum_f # softmax formula # p.shape = (500,10) # softmax such that all probs = 1
  p_by_y = p[np.arange(num_train),y] # p_by_y.shape = (500,) # the probability that we've assigned to the correct answer for each training sample
  loss = np.sum(-np.log(p_by_y) ) # add up negative logs of the probabilities we've assigned to the correct class - function is lower if on average probabilities for right examples are more correct
  
  # compute gradient
  indices = np.zeros(p.shape) # initialize at zero
  indices[np.arange(num_train),y] = 1
  dW = X.T.dot(p - indices)
  
  loss /= num_train # get average loss
  loss += 0.5 * reg * np.sum(W*W) # apply regularization to loss
  dW /= num_train # average the gradient too
  dW += reg * W # regularize gradient too
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

