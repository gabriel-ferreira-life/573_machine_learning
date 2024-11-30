import numpy as np
'''

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.eig(): compute the eigen decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

    def __init__(self, X, n_components):
        '''
        Args:
            X: The data matrix of shape [n_samples, n_features].
            n_components: The number of principal components. A scaler number.
        '''

        self.n_components = n_components
        self.X = X
        self.Up, self.Xp = self._do_pca()

    
    def _do_pca(self):
        '''
        To do PCA decomposition.
        Returns:
            Up: Principal components (transform matrix) of shape [n_features, n_components].
            Xp: The reduced data matrix after PCA of shape [n_samples, n_components].
        '''
        ### YOUR CODE HERE
        # Center the data (X - Xmean)
        Xmean = np.mean(self.X, axis=0)
        X_centered = self.X - Xmean
        
        # Compute the covariance matrix
        Xcov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors and sort them in descending order
        eigenvalues, eigenvectors = np.linalg.eigh(Xcov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the top n_components eigenvectors and project the data
        Up = eigenvectors[:, :self.n_components]
        Xp = np.dot(X_centered, Up)
        self.Xmean = Xmean
        return Up, Xp
        ### END YOUR CODE

    def get_reduced(self):
        '''
        To return the reduced data matrix.
        Args:
            X: The data matrix with shape [n_any, n_features] or None. 
               If None, return reduced training X.
        Returns:
            Xp: The reduced data matrix of shape [n_any, n_components].
        '''
        return self.Xp

    def reconstruction(self, Xp):
        '''
        To reconstruct reduced data given principal components Up.

        Args:
        Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

        Return:
        X_re: The reconstructed matrix of shape [n_samples, n_features].
        '''
        ### YOUR CODE HERE
        # Reconstruct the data and add back the mean
        X_re = np.dot(Xp, self.Up.T) + self.Xmean
        return X_re
        ### END YOUR CODE


def reconstruct_error(A, B):
    '''
    To compute the reconstruction error.

    Args: 
    A & B: Two matrices needed to be compared with. Should be of same shape.

    Return: 
    error: the Frobenius norm's square of the matrix A-B. A scaler number.
    '''
    ### YOUR CODE HERE
    # Compute the Frobenius norm of the difference
    error = np.linalg.norm(A - B, ord='fro')**2
    return error
    ### END YOUR CODE

