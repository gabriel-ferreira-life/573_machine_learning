import math


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
    ### YOUR CODE HERE



    ### END YOUR CODE


def gain(data, feature):
    """Compute the gain of data of splitting by feature.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
        feature: index of feature to split the data

    Returns:
        gain of splitting data by feature
    """
    ### YOUR CODE HERE

    # please call entropy to compute entropy


    ### END YOUR CODE


def get_best_feature(data):
    """Find the best feature to split data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        index of feature to split data
    """
    ### YOUR CODE HERE

    # please call gain to compute gain

    ### END YOUR CODE


def build_tree(data):
    ys = {}
    for x, y in data:
        ys[y] = ys.get(y, 0) + 1
    if len(ys) == 1:
        return list(ys)[0]
    feature = get_best_feature(data)
    subtrees = {}
    ### YOUR CODE HERE

    # please split your data with feature and build sub-trees
    # by calling build_tree recursively

    # sub_tree = build_tree(...)

    ### END YOUR CODE
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
    ### YOUR CODE HERE

    # please call test_data to obtain validation error
    # please call prune_tree recursively for pruning tree

    ### END YOUR CODE
