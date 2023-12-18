import pandas as pd
import dataset


class ethos:
    def __init__(self):
        self.label = {
            0: "Supports",
            1: "Refutes",
            2: "Not enough info",
            3: "Disputed",
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/ethos')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        lines_text = []
        lines_label = []
        for datapoint in dataset["train"]:
            lines_text.append((datapoint["text"]))
            lines_label.append((self.label[datapoint["label"]]))

        dataset_frame = pd.DataFrame({"text": lines_text, "label": lines_label})
        return dataset_frame
