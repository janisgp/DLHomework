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
    
    #Compute loss
    for i in np.arange(0,X.shape[0]):
        #make prediction
        predictionVector = np.zeros(W.shape[1])
        for j in np.arange(0,W.shape[1]): #loop over categories
            for k in np.arange(0,W.shape[0]): #loop over pixel
                predictionVector[j] += W[k,j]*X[i,k]
        #probability normalization factor
        normFact = 0
        for prediction in predictionVector:
            normFact += np.exp(prediction)
        
        loss += -np.log(np.exp(predictionVector[y[i]])/normFact)
        
    #mean over training set
    loss = loss/X.shape[0]
    
    #regularization
    sqSumWeights = 0
    for j in np.arange(0,W.shape[0]):
        for k in np.arange(0,W.shape[1]):
            sqSumWeights += W[j,k]**2
    
    #add regularization to loss
    loss += reg*sqSumWeights
    
    predictionVector = np.matmul(W.T,X[:,:X.shape[1]].T)
    predictionMatrix = np.matmul(W.T,X[:,:X.shape[1]].T)
    
    correctPredictionMatrix = (np.repeat([np.arange(0,W.shape[1])],X.shape[0],axis=0) == y)
    correctPredictionVector = np.sum(np.multiply(correctPredictionMatrix,predictionMatrix),axis=0)
    
    normFactors = np.sum(np.exp(predictionVector),axis=0)
    softmax = np.divide(np.exp(predictionMatrix),normFactors)
    
    #compute gradient
    for i in np.arange(0,W.shape[0]):
        for j in np.arange(0,W.shape[1]):
            for k in np.arange(0,X.shape[0]):
                if j == y[k]:
                    dW[i,j] -= (1-softmax[j,k])*X[k,i]
                elif j != y[k]:
                    dW[i,j] -= -1*softmax[j,k]*X[k,i]
            
            dW[i,j] = dW[i,j]/X.shape[0]
            dW[i,j] += 2*reg*W[i,j]
                
    
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
    
    #########compute loss funtion################################################
    
    #number of training expamples
    N = X.shape[0]
    
    #compute matrix with all predictions Classes x N dim and N dim vector normFactors 
    #and compute mean over elements to ensure numerical stability
    predictionMatrix = np.matmul(W.T,X.T)
    meanprediction = np.mean(predictionMatrix)
    normFactors = np.sum(np.exp(predictionMatrix-meanprediction),axis=0)
    
    #first get Classes x N dim matrix with ones at the position of the correct prediction
    correctPredictionMatrix = (np.repeat([np.arange(0,W.shape[1])],N,axis=0).T == y)
    
    #and then store the probabilities for correct prediction in a N dim vector correctPrediction
    correctPredictionVector = np.sum(np.multiply(correctPredictionMatrix,predictionMatrix),axis=0)
    
    #compute N dim vector with values of the softmax function
    softmax = np.divide(np.exp(correctPredictionVector-meanprediction),normFactors)
    
    #compute overall loss by summing the elements of the softmax vector, dividing by the amounts
    loss = (-1)*np.sum( np.log( softmax ) ) / N + reg*np.sum(np.multiply(W,W))
    
    #########compute gradient####################################################
    
    #compute matrix softmax of Number of classes x training examples
    softmax =  np.divide(np.exp(predictionMatrix-meanprediction),normFactors)
    
    #sum of the partial derivatives of the soft max loss function over the training examples
    dW = (-1)*np.matmul(X.T,(correctPredictionMatrix - softmax).T)/N
    
    #add regularization term
    dW += 2*reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

