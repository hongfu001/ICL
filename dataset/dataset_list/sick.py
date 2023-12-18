import pandas as pd
import dataset

class sick:
    def __init__(self):
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
            }    
    
    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/sick')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        train_lines_text=[]
        train_lines_label=[]

        for datapoint in dataset["train"]:
            train_lines_text.append(("sentence_A: " +datapoint["sentence_A"]+"  sentence_B: " +datapoint["sentence_B"]))
            train_lines_label.append((self.label[datapoint["label"]]))

        dataset_frame=pd.DataFrame({"text":train_lines_text,"label":train_lines_label})

        return dataset_frame
