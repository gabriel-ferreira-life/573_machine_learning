Instruction of programming assignments for COM S 573: Machine Learning


Environment Building
--------------------
Please install python packages tqdm using:
"pip install tqdm" or
"conda install tqdm"


Installation of PyTorch
--------------------------
Please read the instruction in https://pytorch.org/get-started/locally/.


Dataset Descriptions
--------------------
We will use the Cifar-10 Dataset to do the image classification. The 
CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
with 6000 images per class. There are 50000 training images and 10000 
test images.


Assignment Descriptions
-----------------------
There are total three Python files including 'main.py', 'solution.py' and 
'helper.py'. In this assignment, you only need to add your solution in 
'solution.py' file following the given instruction. However, you might need 
to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like 
load data, etc. The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py' and 'helper,py' files. Only try 
to add your code to 'solution.py' file and keep function names and parameters 
unchanged.  


APIs you will need
------------------
nn.Conv2d
nn.MaxPool2d
nn.BatchNorm2d
nn.Linear
nn.ReLU()


