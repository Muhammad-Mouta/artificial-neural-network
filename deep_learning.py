import numpy as np
import helper

# Needs revision after finishing trainer module
#   - Has making an input layer and saving whatever input given to it come in
#     handy? or it turned out to be useless? If it is useless, then, the saved
#     input is redundant and takes up space unnecessarily. Hence this feature
#     would better be removed.

class _NeuralNetworkLayer():
    def __init__(self, W=np.matrix([]), b=np.matrix([]), 
                 Z=np.matrix([]), A=np.matrix([]), g="sigmoid", alpha=0.01):
        """ 
        Initialize a neural network layer object.
        
        Parameters
        ----------
        W : np.matrix, dims=(nl*n[l-1]), where nl is number of nodes of the
            layer and n[l-1] is the number of nodes in the previous layer
        b : np.matrix, dims=(nl*1)
            (W, b) are the weights of the layer.
        Z : np.matrix, dims=(nl*m), where m is the number of examples.
            The linear terms of the layer.
        A : np.matrix, dims=(nl*m)
            The values of the layer. (g(Z), where g is the activation function)
        activation : str, default = "sigmoid"
            The name of the activation function used in the layer, and it could
            be one of ["linear", "sigmoid", "tanh", "relu", "leaky_relu", "softmax"]
        alpha : The hyperparameter used with the leaky relu activation
            
        Returns
        -------
        None
        
        Notes
        -----
        Either (W, b) or (A) must be provided and you could provide all.
        (Z) is optional but if it is provided, the activation function is applied to (Z) and
        the result is saved to (self.A) ignoring any given values for A.
        If only (A) is provided, the object is a mere holder of the given values, 
        and is usually used for the input layer.
        
       """
        # Save the given necessary input (W, b), (A)
        self.W = W
        self.b = b.reshape((-1, 1))
        self.A = A
        
        # Make sure the user gave either (W, b) or (A)
        msg = "You should either provide (W, b) or (A)\nFound\nW = {}, b = {}\nA = {}".format(self.W, self.b, self.A)
        assert ((self.W.size != 0) and (self.b.size != 0)) or (self.A.size != 0), msg
        
        # Assign the given activation to the layer
        self.activation = g
        self.activation_alpha = alpha
        if g == "linear":
            self.g = activations.linear
            self.g_prime = activations.linear_grad
        elif g == "sigmoid":
            self.g = activations.sigmoid
            self.g_prime = activations.sigmoid_grad
        elif g == "tanh":
            self.g = activations.tanh
            self.g_prime = activations.tanh_grad
        elif g == "relu":
            self.g = activations.relu
            self.g_prime = activations.relu_grad
        elif g == "leaky_relu":
            self.g = activations.leaky_relu
            self.g_prime = activations.leaky_relu_grad
        elif g == "softmax":
            self.g = activations.softmax
        else:
            raise AttributeError("Wrong activations, read the documentation for more information!")
        
        # Save the necessary input (Z)
        self.Z = Z
        
        # If the linear terms Z are given, 
        #   Apply the activation function to them and ignore the given A
        if self.Z.size != 0:
            self.A = self.g(self.Z, self.activation_alpha)
        
        # Get th number of nodes of the layer
        if self.A.size != 0:
            self.nl = self.A.shape[0]
        else:
            self.nl = self.W.shape[0]
            
        
    def forward_step(self, X):
        """ 
        Gets the layer's linear terms (Z) and values (A) and 
        updates them in the object.
        
        Parameters
        ----------
        X: np.matrix, dims=(n[l-1]*m)
            The values coming from the previous layer.
            
        Returns
        -------
        None.
        
        # Z: np.matrix dims=(nl*m)
        #     The linear terms of the layer, computed by
        #     Z = np.matmul(self.W, X) + self.b
            
        # A: np.matrix dims=(nl*m)
        #     The values after applying the activation function to Z, computed by
        #     A = sigmoid(Z)
        """
        # Compute the linear term (Z)
        Z = np.matmul(self.W, X) + self.b
        
        # Apply the activation function to get (A)
        A = self.g(Z, self.activation_alpha)
        
        # Save the Z, A of the layer object
        self.Z = Z
        self.A = A
    
    
    def backward_step(self, dA, X):
        """ 
        Computes the grad of the parameters (dW, db) and
        the grad of the previous layer (dX), then updates the layer's (dW, db)
        
        Parameters
        ----------
        dA: np.matrix dims=(nl*m)
            The grad of the layer's values
        X: np.matrix dims=(n[l-1]*m)
            The previous layer's values
        
        Returns
        -------
        dX: np.matrix, dims=(n[l-1]*m)
            The grad of the previous layer's values
        dW: np.matrix, dims=(nl*n[l-1])  
        db: np.matrix, dims=(nl*1)
            (dW, db) are the grad of the layer's weights
        """
        # Get the number of examples
        m = dA.shape[1]
        
        # Compute the linear grad (dZ)
        dZ = np.multiply(dA, self.g_prime(self.Z, self.activation_alpha))
        
        # Compute the activation grad of the previous layer
        dX = np.matmul(self.W.T, dZ)
        
        # Compute the grad of the weights
        dW = (1/m) * np.matmul(dZ, X.T)
        db = (1/m) * np.sum(dZ, axis=1).reshape(-1, 1)
        
        return dX, dW, db
    
    
class NeuralNetwork():
    def __init__(self, W_dict=dict(), b_dict=dict(), n=list(), nx=0, activations=list()):
        """
        Initializes a neural network object with a defined structure.

        Parameters
        ----------
        n : list_of_int, optional
            A list containing the number of nodes in each layer
            where n[0] is the number of input features and n[l] is
            the number of nodes in layer (l).
        W_dict : dict {'str': 'float'}, optional
            A dict containing the weights (W) of the different layers
            where W_dict['W1'] contains the weights of the first layer
            and W_dict['Wl'] contains the weights of layer (l). 
            The weights are of type np.matrix ith dimensions (n[l]*n[l-1]).
        b_dict : dict {'str': 'float'}, optional
            A dict containing the weights (b) of the different layers
            where b_dict['b1'] contains the weights of the first layer
            and b_dict['bl'] contains the weights of layer (l). 
            The weights are of type np.matrix ith dimensions (n[l]*1).
        nx : int, optional
            The number of input features. The default is 0.
        activations : list_of_str, default = "sigmoid"
            A list containing names of the activations for each layer 
            in the network. Supported activations are 
            ["linear", "sigmoid", "tanh", "relu", "leaky_relu"]

        Returns
        -------
        None.

        Notes
        -----
        When you initialize a NeuralNetwork, you have to define its structure.
        This could be done by a number of ways:
            
        (1) The structure could be given explicitly using
            >> nn = NeuralNetwork(n)
            In this case, the number of layers is deduced from the number of
            elements in n, and the weights (W, b) are initialized automatically
            using Random Intialization for (W) and Zero Initialization for (b).
            
        (2) The structure could be given implictly using
            >> nn = NeuralNetwork(W_dict, b_dict)
            In this case, the number of layers is deduced from 
            the number of weight matrices (W) in (W_dict) and the number 
            of nodes in each layer is deduced from the shapes of these
            matrices.
            
            >> nn = NeuralNetwork(W_dict)
            In this case, the number of layers and nodes are deduced the same
            way as above, however, b vectors are zero initialized.
            
            >> nn = NeuralNetwork(b_dict, nx) or nn = NeuralNetwork(b_dict, n)
            In this case the number of layers is deduced from the number of 
            weight vectors (b) within (b_dict) and the number of nodes of each 
            layer except the input layer are deduced from the shape of these 
            vectors.
            The number of input features is either given directly through (nx)
            or given in the list (n) as n[0].
            Notice that if (n) is given, only n[0] is considered to get the 
            number of input features and the rest of the values in the list
            are ignored.
            
        (3) The length of the activations list could be:
            - equal to the number of layers in the network 
              including the i/p layer. In this case, activations[0] is ignored.
            - equal to the number of layers in the network
              excluding the i/p layer. In this case, activations[0] is not ignored,
              and is the activation for layer[1].
        """
        
        if (W_dict != dict() and b_dict != dict()):
            assert len(W_dict) == len(b_dict)
            
            # Get the number of layers excluding the i/p layer
            self.L = len(W_dict)
            
            # Get the activations
            if activations == list():
                activations = ["sigmoid"] * self.L
            else:
                # Make sure that the activations are of size L
                if len(activations) == self.L+1:
                    activations = activations[1:]
                if len(activations) != self.L:
                    raise AttributeError("Wrong activations, read the documentation for more information!")
            
            # Get the number of input features and construct the input layer
            self.n = [np.matrix(W_dict['W1']).shape[1]]
            self.layer = [_NeuralNetworkLayer(A=np.zeros((self.n[0], 1)))]
            
            # Construct the rest of the layers
            for i in range(1, self.L + 1):
                # Cast the given weights to np.matrix
                W, b = np.matrix(W_dict['W' + str(i)], dtype="float64"), np.matrix(b_dict['b' + str(i)], dtype="float64")
                # Assert shapes are ok
                assert W.shape[1] == self.n[i-1], "Wrong shape: W{} ".format(i)
                assert b.shape[0] == W.shape[0], "Wrong shapes: W{}, b{}".format(i, i)
                assert b.shape[1] == 1, "Wrong shape: b{}".format(i)
                # Update the number of nodes in the current layer
                self.n.append(W.shape[0])
                # Construct the layer
                self.layer.append(_NeuralNetworkLayer(W=W, b=b, g=activations[i-1]))
            
        
        elif (W_dict != dict()):
            # Get the number of layers excluding the i/p layer
            self.L = len(W_dict)
            
            # Get the activations
            if activations == list():
                activations = ["sigmoid"] * self.L
            else:
                # Make sure that the activations are of size L
                if len(activations) == self.L+1:
                    activations = activations[1:]
                if len(activations) != self.L:
                    raise AttributeError("Wrong activations, read the documentation for more information!")
            
            # Get the number of input features and construct the input layer
            self.n = [np.matrix(W_dict['W1']).shape[1]]
            self.layer = [_NeuralNetworkLayer(A=np.zeros((self.n[0], 1)))]
            
            # Construct the rest of the layers
            for i in range(1, self.L + 1):
                # Cast W to np.matrix
                W = np.matrix(W_dict['W' + str(i)], dtype="float64")
                # Assert shapes are ok
                assert W.shape[1] == self.n[i-1], "Wrong shape: W{} ".format(i)
                # Get the number of nodes in the current layer
                self.n.append(W_dict['W' + str(i)].shape[0])
                # Zero initialize b
                b = np.zeros((self.n[i], 1))
                # Construct the layer
                self.layer.append(_NeuralNetworkLayer(W=W,b=b, g=activations[i-1]))
            
                
        elif (b_dict != dict()) & (nx > 0 or n != list()):
            if (nx <= 0) and (n[0] <= 0):
                raise AttributeError("If you only provide b_dict, then you must also provide nx or n[0]")
            # Get the number of layers excluding the i/p layer
            self.L = len(b_dict)
            
            # Get the activations
            if activations == list():
                activations = ["sigmoid"] * self.L
            else:
                # Make sure that the activations are of size L
                if len(activations) == self.L+1:
                    activations = activations[1:]
                if len(activations) != self.L:
                    raise AttributeError("Wrong activations, read the documentation for more information!")
            
            # Get the number of input features and construct the input layer
            self.n = [nx if nx > 0 else n[0]]
            self.layer = [_NeuralNetworkLayer(A=np.zeros((self.n[0], 1)))]
            
            # Construct the rest of the layers
            for i in range(1, self.L + 1):
                # Cast b to np.matrix
                b = np.matrix(b_dict['b' + str(i)], dtype="float64")
                # Assert shapes are ok
                assert b.shape[1] == 1, "Wrong shape: b{}".format(i)
                # Get the number of nodes in the current layer
                self.n.append(b.shape[0])
                # Randomly initialize W
                W = helper.W_random_init((self.n[i], self.n[i-1]))
                # Construct the layer
                self.layer.append(_NeuralNetworkLayer(W=W, b=b, g=activations[i-1]))
            
                
        elif n != list():
            # Get the number of layers excluding the i/p layer
            self.L = len(n) - 1
            
            # Get the activations
            if activations == list():
                activations = ["sigmoid"] * self.L
            else:
                # Make sure that the activations are of size L
                if len(activations) == self.L+1:
                    activations = activations[1:]
                if len(activations) != self.L:
                    raise AttributeError("Wrong activations, read the documentation for more information!")
            
            # Save the layers' node counts
            self.n = n
            
            # Construct the input layer
            self.layer = [_NeuralNetworkLayer(A=np.zeros((self.n[0], 1)))]
            
            # Construct the rest of the layers
            for i in range(1, self.L + 1):
                # Randomly initialze W and Zero initialize b
                W, b = helper.W_random_init((self.n[i], self.n[i-1])), np.zeros((n[i], 1))
                # Construct the layer
                self.layer.append(_NeuralNetworkLayer(W=W, b=b, g=activations[i-1]))
        
        
        else:
            raise AttributeError("Wrong initialization, read the documentation for more information!")
            
            
    def for_prop(self, X):
        """
        Performs forward propagation along the Neural Network given input matrix X,
        and updates the values of (Z) and (A) of each layer.

        Parameters
        ----------
        X : array_like dims=(n[0]*m)
            Input features where X[:, i] is the ith input features.

        Returns
        -------
        Y : np.matrix dims=(n[-1]*m)
            Output predictions where Y[:, i] is the ith prediction.

        """
        # Cast X to np.matrix
        X = np.matrix(X)
        
        # Make sure the input shape is ok
        msg = "Wrong input shape: expected {} but found {}".format((self.layer[1].W.shape[1], X.shape[1]), X.shape)
        assert (np.matrix(X).shape[0] == self.layer[1].W.shape[1]), msg
        
        # Save X to layer[0]'s A and delete X
        self.layer[0].A = X
        del X
        
        # Iterate through layers and update their A values
        for l in range(1, self.L+1):
            self.layer[l].forward_step(self.layer[l-1].A)
        
        # The prediction is the last layer's A value
        return self.layer[-1].A
    
    
    def back_prop(self, dAL):
        """
        Back-propagates along the Neural Network to compute the gradients of
        W and b.

        Parameters
        ----------
        dAL : array_like, dims=(n[-1]*m)
            dJ/dAL: partial derivarive of the cost (J) w.r.t the values of
            the last layer (AL).

        Returns
        -------
        dW_dict : dict_of_np.matrix
            A dictionary holding the grads of W, where dW_dict['dWl'] is the
            grads of W of layer[l]
        
        db_dict : dict_of_np.matrix
            A dictionary holding the grads of b, where db_dict['dbl'] is the
            grads of b of layer[l]
        """
        # Cast dAL to np.matrix and rename it to dA
        dA = np.matrix(dAL)
        
        # Make sure the input shape is ok
        msg = "Wrong input shape: expected {} but found {}".format(self.layer[-1].A.shape, dA.shape)
        assert (self.layer[-1].A.shape == dA.shape), msg
        
        # Initialize W_dict and b_dict
        dW_dict, db_dict = dict(), dict()
        
        # Iterate through layers and populate W_dict and b_dict
        for l in reversed(range(1, self.L+1)):
            dA, dW_dict['dW' + str(l)], db_dict['db' + str(l)] = self.layer[l].backward_step(dA, self.layer[l-1].A)
        
        return dW_dict, db_dict
    
    def clear_caches(self):
        """
        Clears the 'Z's and 'A's of the nn (by assigning their values to None)
        to free up space.

        Returns
        -------
        None.

        """
        for l in self.layer:
            l.Z = None
            l.A = None
    
    
class Trainer():
    def __init__(self, alpha=1):
        """
        Initializes a neural network Trainer object

        Parameters
        ----------
        alpha: float, default = 1
            Learning rate of the optimization algorithm

        Returns
        -------
        None.

        """
        self.alpha = alpha
        

    @staticmethod
    def log_reg_cost(Y, Y_hat):
        """
        Computes the logistic regression cost of the predictions using
        cost = (-1/m) * np.sum(np.multiply(Y, np.log(Y_hat)) + np.multiply((1-Y), np.log(1 - Y_hat)))

        Parameters
        ----------
        Y : np.matrix(dtype=float), dims=(n[-1]*m)
            Actual output values.
        Y_hat : np.matrix(dtype=float), dims=(n[-1]*m)
            Predicted output values.

        Returns
        -------
        float
            The logistic rergression cost of the predictions.

        """
        m = Y.shape[1]
        return (-1/m) * np.sum(np.multiply(Y, np.log(Y_hat)) + np.multiply((1-Y), np.log(1 - Y_hat)))
    
    
    @staticmethod
    def log_reg_cost_grad(Y, Y_hat):
        """
        Computes dJ/dAL: partial derivarive of the cost (J) w.r.t the values of
        the last layer (AL).

        Parameters
        ----------
        Y : np.matrix, dims=(n[-1]*m)
            Actual output values.
        Y_hat : np.matrix, dims=(n[-1]*m)
            Predicted output values.

        Returns
        -------
        np.matrix(dtype=float64), dims=(n[-1]*m)
            dJ/dAL.

        """
        return -1 * np.divide(Y, Y_hat) + np.divide(1-Y, 1-Y_hat)
    
        
    def train(self, nn, X, Y):
        """
        Updates the parameters of the given nn using batch gradient descent
        and logistic regression cost function.

        Parameters
        ----------
        nn : deep_learning.NeuralNetwork()
            The NeuralNetwork to be trained.
        X : array_like, dims=(n[0]*m)
            Input features where X[:, i] is the ith input features.
        Y : array_like, dims=(n[-1]*m)
            Output values where Y[:, i] is the ith value.

        Returns
        -------
        None.

        """
        # Cast Y to np.matrix
        Y = np.matrix(Y)
        
        # Compute the predictions using for_prop, no need to cast X because for_prop casts it
        Y_hat = nn.for_prop(X)
        
        # Compute the cost of the predictions
        self.cost = self.log_reg_cost(Y, Y_hat)
        
        # Compute dAL to perform back_prop
        dAL = self.log_reg_cost_grad(Y, Y_hat)
        
        # Perform back_prop to get dW_dict and db_dict
        dW_dict, db_dict = nn.back_prop(dAL)
        
        # Update the parameters using batch gradient descent
        for i in range(len(dW_dict)):
            nn.layer[i+1].W -= self.alpha * dW_dict['dW'+ str(i+1)]
            nn.layer[i+1].b -= self.alpha * db_dict['db'+ str(i+1)]
            
        # Clear the nn caches
        nn.clear_caches()
        
        
class metrics():
    def misclass_count(Y, Y_hat):
        """
        Returns the number of misclassified examples.
        
        Parameters
        ----------
        Y : np.matrix, dims=(n[-1], m)
            The actual output, where c is the number of classes 
            and m is the number of examples.
        Y_hat : np.matrix, dims=(n[-1], m)
            The predicted output, where c is the number of classes 
            and m is the number of examples.
    
        Returns
        -------
        int
           The number of misclassified examples.
    
        """
        m = Y.shape[1]
        return np.sum([0 if np.all(np.equal(Y[:, i], Y_hat[:, i])) else 1 for i in range(m)])
                
        
class activations():
    def linear(Z, alpha=1):
        """
        Computes the element-wise linear function:
            linear(Z, alpha) = alpha * Z

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : float, default = 1
            The slope of the line.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation values of layer[l].

        """
        return alpha * Z
    
    
    def linear_grad(Z, alpha=1):
        """
        Computes the element-wise linear gradient (d(linear(Z))/dZ) of the given matrix:
            linear_grad(Z, alpha) = alpha
        
        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : float, default = 1
            The slope of the line.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation grad values of layer[l].
            
        """
        return alpha * np.ones(Z.shape)
    
    def sigmoid(Z, alpha=None):
        """
        Computes the element-wise sigmoid of the given matrix:
            sigmoid(Z) = 1 / (1 + np.exp(-Z))

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : ignored.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation values of layer[l].

        """
        return 1 / (1 + np.exp(-Z))


    def sigmoid_grad(Z, alpha=None):
        """
        Computes the element-wise sigmoid gradient (d(sigmoid(Z))/dZ) of the given matrix:
            A = deep_learning.activations.sigmoid(Z)
            sigmoid_grad(Z) = np.multiply(A, 1-A)
        
        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : ignored.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation grad values of layer[l].

        """
        A = activations.sigmoid(Z)
        return np.multiply(A, 1-A)
    
    
    def tanh(Z, alpha=None):
        """
        Computes the element-wise tanh of the given matrix:
            tanh(Z) = np.tanh(Z)

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : ignored.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation values of layer[l].

        """
        return np.tanh(Z)
    
    
    def tanh_grad(Z, alpha=None):
        """
        Computes the element-wise tanh gradient d(tanh(Z))/dZ of the given matrix:
            A = ativations.tanh(Z)
            tanh_grad(Z) = 1 - np.power(A, 2)

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : ignored.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation grad values of layer[l].

        """
        A = activations.tanh(Z)
        return 1 - np.power(A, 2)
    
    
    def relu(Z, alpha=None):
        """
        Computes the element-wise ReLU (Rectified Linear Unit) of the given matrix:
            relu(Z) = np.maximum(0, Z)

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : ignored.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation values of layer[l].

        """
        return np.maximum(0, Z)
    
    
    def relu_grad(Z, alpha=None):
        """
        Computes the element-wise ReLU grad d(relu(Z))/dZ of the given matrix:
            Z[Z >= 0] = 1
            Z[Z < 0] = 0
            relu(Z) = Z

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : ignored.

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation grad values of layer[l].

        """
        Z[Z >= 0] = 1
        Z[Z < 0] = 0
        return Z
    
    
    def leaky_relu(Z, alpha=0.01):
        """
        Computes the element-wise leaky ReLU (Rectified Linear Unit) of the given matrix:
            relu(Z) = np.maximum(alpha*Z, Z)

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : float, default=0.01
            The slope of the leakage in the dead zone of the function (zeta < 0)

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation values of layer[l].

        """
        return np.maximum(alpha*Z, Z)
    
    
    def leaky_relu_grad(Z, alpha=0.01):
        """
        Computes the element-wise leaky ReLU grad d(leaky_relu_grad(Z))/dZ of the given matrix:
            Z[Z >= 0] = 1
            Z[Z < 0] = alpha
            leaky_relu_grad = Z

        Parameters
        ----------
        Z : np.matrix(dtype=float), dims=(n[l]*m)
            The linear terms of layer[l].
        alpha : float, default=0.01
            The slope of the leakage in the dead zone of the function (zeta < 0)

        Returns
        -------
        np.matrix(dtype=float), dims=(n[l]*m)
            The activation grad values of layer[l].

        """
        Z[Z >= 0] = 1
        Z[Z < 0] = alpha
        return Z
    
    # Broken    
    # def softmax(Z, alpha=None):
    #     """
    #     Computes the element-wise softmax of the given matrix:
    #         numerator = np.exp(Z)
    #         denominator = np.sum(numerator, axis=0, keep_dims=1)
    #         softmax(Z) = numerator/denominator

    #     Parameters
    #     ----------
    #     Z : np.matrix(dtype=float), dims=(n[l]*m)
    #         The linear terms of layer[l].
    #     alpha : ignored.

    #     Returns
    #     -------
    #     np.matrix(dtype=float), dims=(n[l]*m)
    #         The activation values of layer[l].

    #     """
    #     numerator = np.exp(Z)
    #     denominator = np.sum(numerator, axis=0, keepdims=1)
    #     return numerator/denominator
    
    # def softmax_grad(Z):
    #     """
    #     Computes the element-wise softmax_grad of the given matrix:
    #         A = activations.softmax(Z)
    #         n, m = A.shape
    #         dA = np.zeros((n, m))
    #         for j in range(n):
    #             vec = np.zeros((n, 1))
    #             vec[j, 1] = 1
    #             dA[j, :] = np.multiply(-A[j, :], (A - vec))
    #         softmax_grad(Z) = dA
            
    #     Notice that: 
    #         - dA is a short for dA/dZ = d(softmax(Z))/dZ
    #         - The return matrix is 3-dimensional because for one example
    #         each single element of A has a gradient with all values of Z for 
    #         the same example. 
    #         (i.e. there is d(alpha_1)/d(zeta_1), d(alpha_1)/d(zeta_2), .....
    #          and there is d(alpha_2)/d(zeta_1), d(alpha_2)/d(zeta_2), .....)
        
    #     Parameters
    #     ----------
    #     Z : np.matrix(dtype=float), dims=(n[l]*m)
    #         The linear terms of layer[l].

    #     Returns
    #     -------
    #     np.matrix(dtype=float), dims=(n[l]*n[l]*m)
    #         The activation grad values of layer[l].

    #     """
    #     A = activations.softmax(Z)
    #     n, m = A.shape
    #     dA = np.zeros((n, n, m))
    #     for j in range(n):
    #         vec = np.zeros((n, 1))
    #         vec[j, 0] = 1
    #         # C = (A - vec)
    #         # print(C)
    #         # B = -A[j, :].reshape(1, -1)
    #         # print(B)
    #         # dA[j, :] = C * B
    #         for i in range(m):
    #             dA[j, i, :] = - A[j, i] * (A[:, i] - vec)
    #     return dA