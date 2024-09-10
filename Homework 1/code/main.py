from helper import *
from solution import *


def visualize_data(data_train):
	
	nums = 2
	data_vis = data_train[0:nums,1:]
	[n,d] = data_vis.shape
	w = math.floor(math.sqrt(d))
	img_stack = np.reshape(data_vis, (nums, w, w))
	show_images(img_stack)

	print("Data visualization done.")


def visualize_features(X_train, y_train):

	show_features(X_train[:,1:3],y_train)

	print("Features visualization done.")


def test_accuracy(X_train, y_train, X_test, y_test):
	max_iter = [10, 30, 50, 100, 200]
	for i, m_iter in enumerate(max_iter):
		_, train_acc, test_acc = test_perceptron(m_iter, X_train, y_train, 
												 X_test, y_test)
		print("Case %d: max iteration:%d  train accuracy:%f  test accuracy: %f."
			  %(i+1, m_iter, train_acc, test_acc))

	print("Accuracy testing done.")


def visualize_result(X_train, y_train, X_test, y_test):

	max_iter = 10
	w, _, _ = test_perceptron(max_iter, X_train, y_train, X_test, y_test)
	show_result(X_test[:,1:3], y_test, w)

	print("Result visualization done.")


if __name__ == '__main__':

	traindataloc = "../data/train.txt"
	testdataloc = "../data/test.txt"
	data_train = load_data(traindataloc)
	X_train, y_train = load_features(traindataloc)
	X_test, y_test = load_features(testdataloc)

	## test question (a)
	visualize_data(data_train)
	## test question (b)
	visualize_features(X_train, y_train)
	## test question (c)
	test_accuracy(X_train, y_train, X_test, y_test)
	## test question (d)
	visualize_result(X_train, y_train, X_test, y_test)