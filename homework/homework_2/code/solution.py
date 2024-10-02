from sklearn.svm import SVC
from sklearn import svm

def svm_with_diff_c(train_label, train_data, test_label, test_data):
    '''
    Use different value of cost c to train a svm model. Then apply the trained model
    on testing label and data.
    
    The value of cost c you need to try is listing as follow:
    c = [0.01, 0.1, 1, 2, 3, 5]
    Please set kernel to 'linear' and keep other parameter options as default.
    No return value is needed
    ''' 

    ### YOUR CODE HERE
    print(f"{'-'*95}")
    print("Using different value of cost c to train a svm model. Then apply the trained model on testing label and data.")
    print("")
    # List of Cs to try
    c_values = [0.01, 0.1, 1, 5, 10]
    
    # Loop through each value of C and train an SVM model
    for c in c_values:

        # Build SVM model with linear kernel and cost parameter C
        model = svm.SVC(C=c, kernel='linear')
        
        # Train the model
        model.fit(train_data, train_label)
        
        # Calculate accuracy on the test data
        accuracy = model.score(test_data, test_label)
        
        # Get the number of support vectors
        num_support_vectors = len(model.support_)
        
        # Print the results
        print(f"C={c} | Accuracy={accuracy:.4f} | Support Vectors={num_support_vectors}")
        print("")

    print(f"{'-'*95}")
    print(" ")

    ### END YOUR CODE
    

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    '''
    Use different kernel to train a svm model. Then apply the trained model
    on testing label and data.
    
    The kernel you need to try is listing as follow:
    'linear': linear kernel
    'poly': polynomial kernel
    'rbf': radial basis function kernel
    Please keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE
    print(f"{'-'*95}")
    print("Using different kernel funcitons to train a svm model. Then apply the trained model on testing label and data.")
    print("")
    # List of kernels to try
    kernels = ['linear', 'poly', 'rbf']
    
    # Loop through each kernel type and train an SVM model
    for kernel in kernels:

        # Create the SVM model with the specified kernel
        model = svm.SVC(kernel=kernel)
        
        # Train the model
        model.fit(train_data, train_label)
        
        # Calculate accuracy on the test data
        accuracy = model.score(test_data, test_label)
        
        # Get the number of support vectors
        num_support_vectors = len(model.support_)
        
        # Print the results
        print(f"Kernel={kernel} | Accuracy={accuracy:.4f} | Support Vectors={num_support_vectors}")
        print("")
    print(f"{'-'*95}")
    ### END YOUR CODE
