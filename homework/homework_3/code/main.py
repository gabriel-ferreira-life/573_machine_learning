import copy
import random
from helper import read_data
from solution import test_data, build_tree, prune_tree


def test_tree_accuracy(data):
    random.seed(1)
    print('=' * 58)
    trainlen = len(data['train'])

    train_data = data['train']
    random.shuffle(train_data)

    tree = build_tree(train_data[:int(trainlen * 0.8)])
    print(
        'Validate accuracy on tree without pruning ========>',
        test_data(tree, train_data[int(trainlen * 0.8):]))
    Ptree = prune_tree(
        copy.deepcopy(tree), train_data[int(trainlen * 0.8):])
    print(
        'Validate accuracy on tree with pruning ===========>',
        test_data(Ptree, train_data[int(trainlen * 0.8):]))
    print(
        'Test accuracy on tree without pruning ============>',
        test_data(tree, data['test']))
    print(
        'Test accuracy on tree with pruning ===============>',
        test_data(Ptree, data['test']))
    print('Tree size without pruning ========================> %6d' % (
        tree.size))
    print('Tree size with pruning ===========================> %6d' % (
        Ptree.size))
    print('Tree depth without pruning =======================> %6d' % (
        tree.depth))
    print('Tree depth with pruning ==========================> %6d' % (
        Ptree.depth))
    print('=' * 58)


if __name__ == '__main__':
    data = read_data(dataloc='../data/car_evaluation.csv')
    test_tree_accuracy(data)
