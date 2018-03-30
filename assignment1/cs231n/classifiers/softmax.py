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
  pass
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
  N = X.shape[0]
  C = W.shape[1]
  scores = np.dot(X, W)
  shift_scores = scores - np.max(scores, axis=1)[..., np.newaxis]
  exp_scores = np.exp(shift_scores)
  softmax_scores = exp_scores / np.sum(exp_scores, axis=1)[..., np.newaxis]

  correct_class_prob = softmax_scores[np.arange(N),y]
  loss = np.sum(-np.log(correct_class_prob))

  dScore = softmax_scores
  dScore[np.arange(N), y] -= 1
  dW = np.dot(X.T, dScore)
  dW /= N
  dW += 2 * reg * W

  loss /= N
  loss += reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized_sol(W, X, y, reg):
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

  # Calculate scores and numeric stability fix.
  scores = np.dot(X, W)
  shift_scores = scores - np.max(scores, axis=1)[...,np.newaxis]

  # Calculate softmax scores.
  softmax_scores = np.exp(shift_scores)/ np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]

  # Calculate dScore, the gradient wrt. softmax scores.
  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1

  # Backprop dScore to calculate dW, then average and add regularisation.
  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += 2*reg*W

  # Calculate our cross entropy Loss.
  correct_class_scores = shift_scores[np.arange(num_train), y]  # Size N vector
  loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
  loss = np.sum(loss)

  # Average our loss then add regularisation.
  loss /= num_train
  loss += reg * np.sum(W*W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

if __name__ == "__main__":
  N = 2000
  D = 250
  C = 100

  X = np.random.rand(N, D)
  W = np.random.rand(D, C)
  y = np.random.randint(0, C, N)
  reg = 2.0

  (loss, dW) = softmax_loss_vectorized(W, X, y, reg)
  (loss_sol, dW_sol) = softmax_loss_vectorized_sol(W, X, y, reg)

  assert np.allclose(loss, loss_sol, rtol=1e-05, atol=1e-06)
  assert np.allclose(dW, dW_sol, rtol=1e-05, atol=1e-06)

  print ("Pass")