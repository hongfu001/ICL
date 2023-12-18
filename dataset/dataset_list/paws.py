import pandas as pd
import dataset

class paws:
    def __init__(self):
        self.label = {
            0: "not_duplicate",
            1: "duplicate",
            }    
    
    
    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/paws')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        lines_text=[]
        lines_label=[]
        for datapoint in dataset["train"]:
            lines_text.append(("sentence 1: " +datapoint["sentence1"]+"sentence 2: " +datapoint["sentence2"]))
            lines_label.append((self.label[datapoint["label"]]))
    
        dataset_frame=pd.DataFrame({"text":lines_text,"label":lines_label})
        return dataset_frame