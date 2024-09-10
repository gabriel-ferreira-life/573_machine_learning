import numpy as np
import sys
from helper import *
import os


def show_images(data, save=True):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE
    directory = "../output/images/"
    for i in range(0, data.shape[0]):
        
        image = data[i]
        plt.imshow(image, cmap='gray')
        plt.title(f'Image {i+1}')

        if save == True:
            # Check if directory exists, if not, create it
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Saving figure
            plt.savefig(f'{directory}image_{i+1}.png')
        else:
            pass
        
        plt.show()

    ### END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
    directory = "../output/images/"
    # Creating a scatter plot
    plt.figure(figsize=(8, 5))

    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='r', marker='*', label='Label 1' if (i==0 or i==1) else "")

        if y[i] == 5:
            plt.scatter(X[i, 0], X[i, 1], c='b', marker='+', label='Label 5' if (i==0 or i==1) else "")

    plt.xlabel('Sys')
    plt.ylabel('Intense')
    plt.title("2-D Scatter Plot")
    plt.legend()


    if save == True:
        # Check if directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Saving figure
        plt.savefig(f'{directory}image_{i+1}.png')
    else:
        pass

    plt.show()
    


    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        

        # After implementation, assign your weights w to self as below:
        # self.W = w
        
        ### END YOUR CODE
        
        return self

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE



        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE



        ### END YOUR CODE




def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    ### YOUR CODE HERE



    ### END YOUR CODE



def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()

    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return W, train_acc, test_acc