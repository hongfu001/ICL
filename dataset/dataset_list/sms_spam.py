import pandas as pd
import dataset

class sms_spam:
    def __init__(self):
        self.label = {
            0:"ham",
            1:"spam",
            }    
    
    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/sms_spam')
    
    def dataset_to_DataFrame(self):
        dataset=self.load_dataset()
        lines_text=[]
        lines_label=[]
        for datapoint in dataset["train"]:
            lines_text.append((datapoint["sms"]))
            lines_label.append((self.label[datapoint["label"]]))
    
        dataset_frame=pd.DataFrame({"text":lines_text,"label":lines_label})
        return dataset_frame
