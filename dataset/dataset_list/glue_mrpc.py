import pandas as pd
import dataset

class glue_mrpc:
    def __init__(self):
        self.label = {
            0: "not_equivalent",
            1: "equivalent",
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/mrpc')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        train_lines_text = []
        train_lines_label = []

        for datapoint in dataset["train"]:
            train_lines_text.append("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"])
            train_lines_label.append((self.label[datapoint["label"]]))


        dataset_frame = pd.DataFrame({"text": train_lines_text, "label": train_lines_label})

        return dataset_frame

