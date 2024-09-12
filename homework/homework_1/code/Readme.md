Instruction of programming assignments for Homework 1

We will use Python for all assignments in this course. Specifically, we will use a few popular libraries (numpy, matplotlib, math) for scientific computing and plotting.

We assume that many of you have already had some basic experience with Python and Numpy. We also provide a basic guide to learn Python and Numpy in case you are new to them.


Setup
-----
Download and install Anaconda with Python 3.6 version:
- Download Anaconda at this website: https://www.anaconda.com/download/
- Install Python 3.6 version (not Python 2.7)
Anaconda will manage all the Python libraries we need. 

Start programming:
Anaconda and Spyder are recommended to start your programming exercise.


Python & Numpy Tutorial
-----------------------
- Official Python tutorial: https://docs.python.org/3/tutorial/
- Official Numpy tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
- Good tutorial sources: http://cs231n.github.io/python-numpy-tutorial/ 


Dataset Descriptions
--------------------
We will use part of the MNIST dataset for all the assignments. All the data files will be in 'data' folder named by 'train.txt' and 'test.txt'. Here are the details of the dataset information for this assignment:
		     samples   image size   labels 
Training Dataset      1561       16*16      1 or 5
Testing Dataset        424       16*16      1 or 5

We've already extracted two features as discussed in the class so that you can directly use these features as your input. 
feature1: symmetry
feature2: average intensity
Here are the details of the feature information:
		    samples   feature numbers   labels 
Training Dataset     1561           2           1 or 5
Testing Dataset       424           2           1 or 5


Assignment Descriptions
-----------------------
There are three Python files inside the 'code' folder, 'main.py', 'solution.py' and 'helper.py'. For all the assignments, you will only need to write your code in 'solution.py' file following the given instruction. You might need to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like data loading, features extraction, etc. The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py' and 'helper.py' files. Only write your code in 'solution.py' file and keep function names and parameters unchanged.




