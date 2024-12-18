"""
1. Start at input layer, forward propagate the patterns of the training data through 
the network to generate an output
2. Based on the network's output, we calculate the loss that we want to minimize
using a loss function
3. Backpropogate the loss, find it's derivative with repect to each weight and bias
unit in the network, and update the model

repeat steps 1-3 for a specified number of epochs

use forward propogation to calculatae the network output and apply a threshold function
to obtain the predicted class labels using one-hot encoding
"""
import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
    

def one_hot_encoding(y, num_classifications):
    """
    one hot encode y to allow for more than two classifications
    """
    one_hot = np.zeros((y.shape[0], num_classifications))
    # create a n x c matrix of zeros
    # where n is number of examples in the training set
    # c = num_classifications

    for i, y_val in enumerate(y):
        one_hot[i, y_val] = 1

    return one_hot


class MLP:
    def __init__(self, num_f, num_c, num_h=50, random_seed=42):
        super().__init__()
        self.num_c = num_c

        rng = np.random.RandomState(random_seed)
        

        self.w_h = rng.randn(num_h, num_f) / 10.
        self.b_h = np.zeros(num_h)
        

        self.w_out = rng.randn(num_c, num_h) / 10.
        self.b_out = np.zeros(num_c)



    def forward(self, x):
        """
        forward

        Zh = Xin * Wh.T + bh
            ---- Xin is a nxm feature matrix where n is the number of of examples in the training dataset
            ---- and m is the number of features
            ---- Wh is a dxm weight matrix where d is the number of units in the hidden layer
            ---- bh is a 1xd vector of bias units, one bias unit per hidden node
            ---- Zh is nxd

        Ah = sigmoid(Zh)
            ---- nxd matrix

        Zout = Ah * Wout.T + bout
            ---- Ah is nxd
            ---- Wout is a txd matrix where t is the number of output units
            ---- bout is a t diminsional bias vector
            ---- Zout is nxt
        
        Aout = sigmoid(Zout)
            ---- Aout is nxt

        """
        z_h = np.dot(x, self.w_h.T) + self.b_h
        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, self.w_out.T) + self.b_out
        a_out = sigmoid(z_out)

        return a_h, a_out


    def backward(self, x, y, a_h, a_out):
        """
        backpropogation 
        calculate gradient of loss with respect to the weight and bias parameters  
        """
        y_one_hot = one_hot_encoding(y, self.num_c)

        dl__da_out = 2. * (a_out - y_one_hot) / y.shape[0]
        da_out__dz_out = a_out * (1. - a_out)

        delta_out = dl__da_out * da_out__dz_out

        dl__dw_out = np.dot(delta_out.T, a_h)
        dl__db_out = np.sum(delta_out, axis=0)

        #
        delta_h = np.dot(delta_out, self.w_out)

        da_h__dz_h = a_h * (1. - a_h)

        dl__dw_h = np.dot((delta_h * da_h__dz_h).T, x)
        dl__db_h = np.sum((delta_h * da_h__dz_h), axis=0)

        return (dl__dw_out, dl__db_out, dl__dw_h, dl__db_h)



    
