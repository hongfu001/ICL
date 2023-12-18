from algorithms.train.cross_entropy import CrossEntropy


def create_train_alg(args, net , device, train_loader):
    global alg_obj
    if args.alg_train == "CrossEntropy":
        alg_obj = CrossEntropy(args, net, device, train_loader)

    else:
        alg_obj = False
    return alg_obj

        
