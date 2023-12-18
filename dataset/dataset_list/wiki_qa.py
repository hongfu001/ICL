import pandas as pd
import datasets


class wiki_qa:
    def __init__(self):
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
            }    
    
    @staticmethod
    def load_dataset():
        return datasets.load_dataset('/data/home/llm_dev/gaohongfu/dataset/wiki_qa')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        train_lines_text=[]
        train_lines_label=[]
        
        for datapoint in dataset["train"]:
            train_lines_text.append(("question: " + datapoint["question"] + " [SEP] answer: " + datapoint["answer"]))
            train_lines_label.append((self.label[datapoint["label"]]))

        dataset_frame=pd.DataFrame({"text":train_lines_text,"label":train_lines_label})

        return dataset_frame
