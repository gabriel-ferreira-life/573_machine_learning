import numpy as np
import sys
from helper import *


def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE
    
    for i in range(0, data.shape[0]):
        
        image = data[i]
        plt.imshow(image, cmap='gray')
        plt.title(f'Image {i+1}')
        plt.savefig(f'image_{i+1}.png')
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

    # Creating a scatter plot
    plt.figure(figsize=(8, 5))

    for i in range(len(X)):
        if y[i] == 1.0:
            plt.scatter(X[i, 0], X[i, 1], c='r', marker='*', label='Label 1' if (i==0 or i==1) else "")

        elif y[i] == 5.0:
            plt.scatter(X[i, 0], X[i, 1], c='b', marker='+', label='Label 5' if (i==0 or i==1) else "")


    plt.xlabel('Sys')
    plt.ylabel('Intense')
    plt.title('2-D Scatter Plot')
    plt.legend()

    if save==True:
        plt.savefig(f'2D_scatterplot.png')
    else:
        pass

    plt.show()

    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.W = None 

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE

        
        
        # Parameters
        max_iter = self.max_iter
        self.W = np.zeros(X.shape[1] + 1)
        
        
        for _ in range(max_iter):
            
#             print(self.W)
            
            for xi, target in zip(X, y):
                
                
                xi = np.insert(xi, 0, 1, axis=0)

                prediction = np.dot(xi, self.W)
                prediction_sign = np.where(prediction >= 0.0, 1, -1)
#                 print("xi: ", xi, " W: ", self.W, " prediction: ", prediction, " prediction_sign: ", prediction_sign)
    
                self.W += (target - prediction_sign) * xi


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

        X = np.insert(X, 0, 1, axis=1)
        # Compute the dot product and apply the sign function
        prediction = np.where(np.dot(X, self.W) >= 0.0, 1, -1)

        return prediction

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
        # Predict the class labels for the samples in X
        predictions = self.predict(X)

        # Compare predictions with the actual labels and calculate accuracy
        accuracy = np.mean(predictions == y)
        
        return accuracy
        ### END YOUR CODE




def show_result(X, y, W, save=True):
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

    # Creating a scatter plot
    plt.figure(figsize=(8, 5))

    for i in range(len(X)):
        if y[i] == 1.0:
            plt.scatter(X[i, 0], X[i, 1], c='r', marker='*', label='Label 1' if (i==0 or i==1) else "")

        elif y[i] == -1.0:
            plt.scatter(X[i, 0], X[i, 1], c='b', marker='+', label='Label 5' if (i==0 or i==1) else "")


    # Plot the decision boundary
    x1_values = X[:,0]
    x2_values = - (W[1] / W[2]) * x1_values - (W[0] / W[2])
    plt.plot(x1_values, x2_values, label='Decision Boundary')

    # Set plot limits
    plt.xlim(min(X[:, 0]), max(X[:, 0]))
    plt.ylim(min(X[:, 1]), max(X[:, 1]))

    # Adding labels and title
    plt.xlabel('Sys')
    plt.ylabel('Intense')
    plt.title('Perceptron Results')
    plt.legend()
    
    if save==True:
        plt.savefig(f'2D_scatterplot_decision_b.png')
    else:
        pass
    
    plt.show()


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