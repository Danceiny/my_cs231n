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
    dW = np.zeros_like(W) # dW[D,C]

    
#    for i in xrange(num_train):
#        scores = X[i].dot(W) # X[1,D] dot W[D,C] = scores[1,C]
#        correct_class_score = scores[y[i]]
#        for j in xrange(num_classes):
#            if j==y[i]:
#                continue
#            prob_numerator = np.exp(correct_class_score)
#            prob_denominator += np.exp(scores[j])
#        loss = -np.log(prob_numerator / prob_denominator)
#        dW
#        
    dW_each = np.zeros_like(W)
    num_train,dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W) # f[N,C]
    # considering the numeric stability
    f_max = np.reshape(np.max(f,axis=1),(num_train,1)) # f_max[N,1]
    #prob = np.exp(f-f_max) / np.sum(np.exp(f-f_max),axis=1,keepdims=True) # prob[N,C]
    prob = np.exp(f) / np.sum(np.exp(f),axis=1,keepdims=True) # prob[N,C]
    y_correct = np.zeros_like(prob)
    y_correct[np.arange(num_train),y] = 1.0
    for i in xrange(num_train):
        for j in xrange(num_class):
            loss += -(y_correct[i,j] * np.log(prob[i,j]))
            dW_each[:,j] = -(y_correct[i,j] - prob[i,j]) * X[i,:]
        dW += dW_each
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train
    dW += reg *W
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
    
    num_train,dim = X.shape

    f = X.dot(W)    # X[N,D] dot W[D,C] = f[N,C]
    # considering the numeric stability
    f_max = np.reshape(np.max(f,axis=1),(num_train,1)) # f_max[N,1]
    
    prob = np.exp(f-f_max) / np.sum(np.exp(f-f_max),axis=1,keepdims=True) # prob[N,C]
    y_correct = np.zeros_like(prob)
    y_correct[np.arange(num_train),y] = 1.0
    loss += -np.sum(y_correct * np.log(prob)) / num_train + 0.5 * reg * np.sum(W*W)
    dW += -np.dot(X.T,y_correct-prob) / num_train + reg * W
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

