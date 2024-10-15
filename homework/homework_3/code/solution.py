import math
import numpy as np

class Tree(object):

    def __init__(self, feature=None, ys={}, subtrees={}):
        self.feature = feature
        self.ys = ys
        self.subtrees = subtrees

    @property
    def size(self):
        size = 1
        for subtree in self.subtrees.values():
            if type(subtree) == int:
                size += 1
            else:
                size += subtree.size
        return size

    @property
    def depth(self):
        max_depth = 0
        for subtree in self.subtrees.values():
            if type(subtree) == int:
                cur_depth = 1
            else:
                cur_depth = subtree.depth
            max_depth = max(cur_depth, max_depth)
        return max_depth + 1
    
def entropy(data):
    """Compute entropy of data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        entropy of data (float)
    """

    # Parse data
    y = np.array([item[1] for item in data])

    # Lenth of the target
    len_y = len(y)

    # Frequency of each label
    labels, counts = np.unique(y, return_counts=True)

    # Compute entropy using label probabilities
    entropy = 0
    for count in counts:
        probability = count / len_y
        if probability > 0:
            entropy -= probability * np.log2(probability)

    return entropy

def gain(data, feature):
    """Compute the gain of data of splitting by feature.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
        feature: index of feature to split the data

    Returns:
        gain of splitting data by feature
    """

    # Base Entropy
    base_entropy = entropy(data)

    # Unique values of the given feature
    values = set([x[feature] for x, y in data])

    # Calculate feature weighted entropy
    weighted_entropy = 0
    for value in values:
        subset = [(x, y) for x, y in data if x[feature] == value]
        subset_entropy = entropy(subset)
        subset_probability = len(subset)/ len(data)
        weighted_entropy += subset_probability*subset_entropy
    
    # Return Information Gain of the feature
    return base_entropy - weighted_entropy

def get_best_feature(data):
    """Find the best feature to split data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        index of feature to split data
    """
    # Initialize features
    best_feature = -1
    max_gain = np.float64('-inf')

    # Loop through the range of columns in the dataset
    for col in range(len(data[0][0])):

        # Compute the feature gain
        feature_ig = gain(data,col)

        # Compare current feature_ig with the max_gain
        if feature_ig > max_gain:
            max_gain = feature_ig
            best_feature = col

    # Return the feature with the highest IG
    return best_feature


def build_tree(data):
    ys = {}
    for x, y in data:
        ys[y] = ys.get(y, 0) + 1
    if len(ys) == 1:
        return list(ys)[0]
    feature = get_best_feature(data)
    subtrees = {}

    # Unique values of the selected feature
    values = set([x[feature] for x, y in data])

    # Split data and build subtrees recursively for each value of the feature
    for value in values:
        subset = [(x, y) for x, y in data if x[feature] == value]
        subtrees[value] = build_tree(subset)

    return Tree(feature, ys, subtrees)


def test_entry(tree, entry):
    x, y = entry
    if type(tree) == int:
        return tree, y
    if x[tree.feature] not in tree.subtrees:
        return tree, max([(value, key) for key, value in tree.ys.items()])[1]
    return test_entry(tree.subtrees[x[tree.feature]], entry)


def test_data(tree, data):
    count = 0
    for d in data:
        y_hat, y = test_entry(tree, d)
        count += (y_hat == y)
    return round(count / float(len(data)), 4)

def prune_tree(tree, data):
    """Find the best feature to split data.

    Args:
        tree: a decision tree to prune
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        a pruned tree
    """
    # Check if the current tree node is a leaf node
    if isinstance(tree, int):
        return tree
    
    # Determine what to do if there's no data reaching the current node
    if not data:
        leaf_label = max(tree.ys, key=tree.ys.get) # Majority classs
        return leaf_label

    # Prune each subtree recursively
    for value, subtree in tree.subtrees.items():
        data_subtree = [(x, y) for x, y in data if x[tree.feature] == value]
        tree.subtrees[value] = prune_tree(subtree, data_subtree)

    # Attempt to collapse this node
    leaf_label = max(tree.ys, key=tree.ys.get)
    collapsed_accuracy = sum(1 for x, y in data if y == leaf_label) / len(data)
    current_accuracy = test_data(tree, data)

    # Only prune if collapsed accuracy improves by at least delta
    if collapsed_accuracy > current_accuracy:
        return leaf_label
    else:
        return tree