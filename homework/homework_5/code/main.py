from solution import LeNet, get_args
from helper import set_random, load_data, train, test


if __name__ == '__main__':
    args = get_args()
    set_random(args.seed)
    trainloader, testloader = load_data(args.batch)
    net = LeNet(args)
    train(net, trainloader, args.num_epochs)
    test(net, testloader)
