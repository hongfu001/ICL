import pandas as pd
import datasets


class climate:
    def __init__(self):
        self.hf_identifier = "climate_fever"
        self.task_type = "classification"
        self.label = {
            0: "Supports",
            1: "Refutes",
            2: "Not enough info",
            3: "Disputed",
        }

    @staticmethod
    def load_dataset():
        return datasets.load_dataset('/data/home/llm_dev/gaohongfu/dataset/climate_fever')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        lines_id = []
        lines_text = []
        lines_label = []
        for datapoint in dataset["test"]:
            lines_id.append((datapoint["claim_id"]))
            lines_text.append((datapoint["claim"]))
            lines_label.append((self.label[datapoint["claim_label"]]))

        dataset_frame = pd.DataFrame({"id":lines_id, "text": lines_text, "label": lines_label,"options": str(self.label)})
        return dataset_frame
