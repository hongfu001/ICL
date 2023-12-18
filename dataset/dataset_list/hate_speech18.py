import pandas as pd
import dataset


class hate_speech18:
    def __init__(self):
        self.label = {
            0: "noHate",
            1: "hate",
            2:"relation",
            3:"idk/skip"
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/hate_speech18')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        lines_text = []
        lines_label = []
        for datapoint in dataset["train"]:
            lines_text.append((datapoint["text"]))
            lines_label.append((self.label[datapoint["label"]]))

        dataset_frame = pd.DataFrame({"text": lines_text, "label": lines_label})
        return dataset_frame