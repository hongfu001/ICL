import pandas as pd
import datasets

class imdb:
    def __init__(self):
        self.label = {
            0: "negative",
            1: "positive",
            }    
    
    @staticmethod
    def load_dataset():
        return datasets.load_dataset('/data/home/llm_dev/gaohongfu/dataset/imdb')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        train_lines_text=[]
        train_lines_label=[]
        
        
        for datapoint in dataset["train"]:
            train_lines_text.append((datapoint["text"]))
            train_lines_label.append((self.label[datapoint["label"]]))


        dataset_frame=pd.DataFrame({"text":train_lines_text,"label":train_lines_label})

        return dataset_frame