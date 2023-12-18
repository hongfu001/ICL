import pandas as pd
import dataset
class health_fact:
    def __init__(self):
        self.label = {
            -1:"NULL",
            0:"false",
            1:"mixture",
            2:"true",
            3:"unproven",
            }    

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/health_fact')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        train_lines_text=[]
        train_lines_label=[]
        
        
        for datapoint in dataset["train"]:
            train_lines_text.append((datapoint["claim"].strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")))
            train_lines_label.append((self.label[datapoint["label"]]))

        dataset_frame=pd.DataFrame({"text":train_lines_text,"label":train_lines_label})

        return dataset_frame

dataset=health_fact()
print(dataset.load_dataset())
