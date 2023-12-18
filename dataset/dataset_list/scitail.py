import pandas as pd
import dataset


class scitail:
    def __init__(self):
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/scitail')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        lines_text = []
        lines_label = []
        for datapoint in dataset["dgem_format"]:
            lines_text.append("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"])
            lines_label.append((self.label[datapoint["label"]]))

        dataset_frame = pd.DataFrame({"text": lines_text, "label": lines_label})
        return dataset_frame
