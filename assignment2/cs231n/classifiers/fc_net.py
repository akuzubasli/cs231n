from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        std = weight_scale
        self.params["W1"] = std * np.random.randn(input_dim, hidden_dim) 
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = std * np.random.randn(hidden_dim, num_classes) 
        self.params["b2"] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Unpacking self.params
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        num_train = X.shape[0]
        # Compute score using affine_forward(x, w, b) - relu_forward(x)
        # intermediates
        z1, cache = affine_forward(X, W1, b1) 
        x2, cache = relu_forward(z1) # output of hidden layer -1 / input of second layer
        # score
        scores, cache = affine_forward(x2, W2, b2)
    
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################      
        # Compute log probs and loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        loss = data_loss + reg_loss
        
        # Compute grads without reg
        dx2, dW2, db2 = affine_backward(dscores, cache = (x2, W2, b2))
        dz1 = relu_backward(dx2, cache = z1)
        dX, dW1, db1 = affine_backward(dz1, cache = (X, W1, b1))
        
        # Add reg gradients to grads
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        
        # Above can also be calculated (faster) through affine_relu_forward/backward fromn layer_utils 
        
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        # print(self.num_layers)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        num_layers = self.num_layers
        std = weight_scale
        for i in range(num_layers-1): # Notice: Actual layer number is i+1
            self.params["W"+ str(i+1)] = std * np.random.randn(input_dim if i==0 else hidden_dims[i-1] , hidden_dims[i]) 
            self.params["b"+ str(i+1)] = np.zeros(hidden_dims[i])
            self.params["gamma"+str(i+1)] = np.ones(hidden_dims[i])
            self.params["beta"+str(i+1)] = np.zeros(hidden_dims[i])
        self.params["W"+ str(num_layers)] = std * np.random.randn(hidden_dims[num_layers-1-1], num_classes) 
        self.params["b"+ str(num_layers)] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers-1)]

        # Cast all parameters to the correct datatype
        # print(self.params.items())
        for k, v in self.params.items():
            # print(self.params[k])
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        num_layers = self.num_layers
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        W, b, gamma, beta, caches= {}, {}, {}, {}, {}
        layer_inputs, affine_out, b_out, relu_out, dropout_out  = {}, {}, {}, {}, {}
        for i in range(1, num_layers):
            # Unpacking self.params
            W[str(i)] = self.params["W"+ str(i)] # store W1 as W["1"]
            b[str(i)] = self.params["b"+ str(i)]
            W[str(num_layers)] = self.params["W"+ str(num_layers)]
            b[str(num_layers)] = self.params["b"+ str(num_layers)]
            gamma[str(i)] = self.params["gamma"+ str(i)] # store W1 as W["1"]
            beta[str(i)] = self.params["beta"+ str(i)]
            
            # Compute score using affine - batch_norm - relu - dropout
            # affine_forward(x, w, b): return out = scores, cache = (x, w, b)
            layer_inputs["1"] = X
            affine_out[str(i)], _= affine_forward(layer_inputs[str(i)], W[str(i)], b[str(i)])
            
            # batchnorm_forward(x, gamma, beta, bn_param): return out, cache = intermediates
            b_out[str(i)] = affine_out[str(i)] # By-pass 
            if self.use_batchnorm:
                b_out[str(i)], caches["batch_cache"+str(i)] = batchnorm_forward(affine_out[str(i)], 
                                                                                gamma[str(i)],
                                                                                beta[str(i)],
                                                                                self.bn_params[i-1])
         
            # relu_forward(x): return dx, cache = x
            relu_out[str(i)], _ = relu_forward(b_out[str(i)]) # output of hidden layer -1 / input of second layer     
        
            # dropout_forward(x, dropout_param): return out, cache = (dropout_param, mask)
            dropout_out[str(i)] = relu_out[str(i)] # By-pass
            if self.use_dropout:
                dropout_out[str(i)], caches["dropout_cache"+str(i)] = dropout_forward(relu_out[str(i)], self.dropout_param)
            
            layer_inputs.update({str(i+1) : dropout_out[str(i)]})
        # calculate scores
        layer_inputs[str(num_layers)] = dropout_out[str(num_layers-1)]
        scores, caches[str(num_layers)] = affine_forward(layer_inputs[str(num_layers)], 
                                                        W[str(num_layers)], b[str(num_layers)])

        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Compute log probs and loss

        data_loss, dscores = softmax_loss(scores, y)
        
        sum_W_sq = 0
        for i in range(1, num_layers+1):
          sum_W_sq += np.sum(W[str(i)]*W[str(i)])
        reg_loss = 0.5 * self.reg * sum_W_sq
              
        loss = data_loss + reg_loss
        
        dW, db, dgamma, dbeta = {}, {}, {}, {}
        dlayer_inputs, daffine_out, db_out, drelu_out, ddropout_out  = {}, {}, {}, {}, {}
       
        # last layer gradient        
        dlayer_inputs[str(num_layers)], dW[str(num_layers)], db[str(num_layers)] = affine_backward(dscores, 
                                                                                                   cache = (layer_inputs[str(num_layers)], 
                                                                                                            W[str(num_layers)], 
                                                                                                            b[str(num_layers)]))
        dW[str(num_layers)] += self.reg * W[str(num_layers)]
        grads["W"+str(num_layers)] = dW[str(num_layers)]
        grads["b"+str(num_layers)] = db[str(num_layers)]                                                                                                            
        
        for i in range(num_layers-1, 0, -1):
          ddropout_out[str(i)] = dlayer_inputs[str(i+1)]
          drelu_out[str(i)] = ddropout_out[str(i)] # By-pass if no dropout
          # dropout_backward(dout, cache = (dropout_param, mask): return dx 
          if self.use_dropout:
            drelu_out[str(i)] = dropout_backward(ddropout_out[str(i)], caches["dropout_cache"+str(i)])
          # relu_backward(dout, cache = x): return dx
          db_out[str(i)] = relu_backward(drelu_out[str(i)], b_out[str(i)])
                    
          #No batchnorm
          daffine_out[str(i)] = db_out[str(i)] # By-pass if no batchnorm
          # batchnorm_backward(dout, cache): return dx, dgamma, dbeta
          dgamma[str(i)], dbeta[str(i)] = 0, 0
          if self.use_batchnorm:
            daffine_out[str(i)], dgamma[str(i)], dbeta[str(i)] = batchnorm_backward_alt(db_out[str(i)], caches["batch_cache"+str(i)])
          ############
          
          dlayer_inputs[str(i)], dW[str(i)], db[str(i)] = affine_backward(daffine_out[str(i)], (layer_inputs[str(i)], W[str(i)], b[str(i)]))
          
          dW[str(i)] += self.reg * W[str(i)]
          
          grads["W"+str(i)] = dW[str(i)]
          grads["b"+str(i)] = db[str(i)]
          grads["gamma"+str(i)] = dgamma[str(i)]
          grads["beta"+str(i)] = dbeta[str(i)]
        
        

        
        # batchnorm_backward(dout, cache): return dx, dgamma, dbeta
        
        
#         # Compute grads without reg
#        dx2, dW2, db2 = affine_backward(dscores, cache = (x2, W2, b2))
#        dz1 = relu_backward(dx2, cache = z1)
#        dX, dW1, db1 = affine_backward(dz1, cache = (X, W1, b1))
#        
#        # Add reg gradients to grads
#        dW1 += self.reg * W1
#        dW2 += self.reg * W2
#        
#        # Above can also be calculated (faster) through affine_relu_forward/backward fromn layer_utils 
#        
#        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
