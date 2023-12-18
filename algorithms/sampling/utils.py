from algorithms.sampling.random import Random

def create_sample_alg(args, alg_sample):
    global alg_sample_obj
    if alg_sample == "random":
        alg_sample_obj = Random(args)

    return alg_sample_obj