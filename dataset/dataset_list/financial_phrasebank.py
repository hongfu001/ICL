import pandas as pd
import dataset


class financial_phrasebank:
    def __init__(self):
        self.label = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/financial_phrasebank')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        lines_text = []
        lines_label = []
        for datapoint in dataset["train"]:
            lines_text.append((datapoint["sentence"]))
            lines_label.append((self.label[datapoint["label"]]))

        dataset_frame = pd.DataFrame({"text": lines_text, "label": lines_label})
        return dataset_frame
