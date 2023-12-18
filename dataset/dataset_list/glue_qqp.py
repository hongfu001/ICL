import pandas as pd
import dataset

class glue_qqp:
    def __init__(self):
        self.label = {
            0: "not_duplicate",
            1: "duplicate",
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/qqp')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        train_lines_text = []
        train_lines_label = []

        for datapoint in dataset["train"]:
            train_lines_text.append("question 1: " + datapoint["question1"] + " [SEP] question 2: " + datapoint["question2"])
            train_lines_label.append((self.label[datapoint["label"]]))

        dataset_frame = pd.DataFrame({"text": train_lines_text, "label": train_lines_label})

        return dataset_frame
