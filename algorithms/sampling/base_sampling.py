class Sampling:
    def __init__(self, args):
        self.args = args
        self.input = args.input
        self.output = args.output
        self.num_demo = args.num_demo
        self.seed = args.seed

    def sample_demo(self,dataset,sampler):
        select_demo = self.demo_selecting(dataset, sampler) 
        return select_demo
    
    def demo_selecting(self, dataset, sampler):
        pass





