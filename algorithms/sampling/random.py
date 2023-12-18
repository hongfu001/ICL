import pandas as pd

import random

from algorithms.sampling.base_sampling import Sampling


class Random(Sampling):
    def demo_selecting(self, dataset, sampler):
        random.seed(self.seed)
        selectable_example = pd.concat([dataset, sampler]).drop_duplicates(['id'], keep=False)
        
        input_demo=[]
        output_demo=[]
        
        for idx_sam,row_sam in sampler.iterrows():
            selectable_one_example=selectable_example.sample(n=self.num_demo, random_state=idx_sam, axis=0)
            example=''
            for index, row in selectable_one_example.iterrows():
                example = example+"\n Input:"+str(row['text'])+"\n Output:"+str(row['label'])
            example = example+"\n Input:"+str(row_sam['text'])+"\n Output:"
            input_demo.append(example)
            output_demo.append(row_sam['label'])
            
        dataset_final = pd.DataFrame({'text' : input_demo, 'label' : output_demo})
        return dataset_final
                