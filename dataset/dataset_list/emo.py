import pandas as pd
import dataset

class emo:
    def __init__(self):
        self.hf_identifier = "emo"
        self.task_type = "classification"
        self.label = {
            0:"others",
            1:"happy",
            2:"sad",
            3:"angry",
            }    
    
    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/emo')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        train_lines_text=[]
        train_lines_label=[]

        for datapoint in dataset["train"]:
            train_lines_text.append((datapoint["text"]))
            train_lines_label.append((self.label[datapoint["label"]]))

        dataset_frame=pd.DataFrame({"text":train_lines_text,"label":train_lines_label})
    
        return dataset_frame
