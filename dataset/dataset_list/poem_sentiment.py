import pandas as pd
import dataset
class poem_sentiment:
    def __init__(self):
        self.label = {
            0:"negative",
            1:"positive",
            2:"no_impact"
        }
    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/poem_sentiment')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        train_lines_text = []
        train_lines_label = []

        for datapoint in dataset["train"]:
            train_lines_text.append((datapoint["verse_text"]))
            train_lines_label.append((self.label[datapoint["label"]]))


        dataset_frame = pd.DataFrame({"text": train_lines_text, "label": train_lines_label})

        return dataset_frame