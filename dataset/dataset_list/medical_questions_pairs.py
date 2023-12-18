import pandas as pd
import dataset


class medical_questions_pairs:
    def __init__(self):
        self.label = {
            0: "Similar",
            1: "Dissimilar",
        }

    @staticmethod
    def load_dataset():
        return dataset.load_dataset('/data/home/llm_dev/gaohongfu/dataset/medical_questions_pairs')

    def dataset_to_DataFrame(self):
        dataset = self.load_dataset()
        lines_text = []
        lines_label = []
        for datapoint in dataset["train"]:
            lines_text.append(("Question 1: " +datapoint["question_1"]+"Question 2: " +datapoint["question_2"]))
            lines_label.append((self.label[datapoint["label"]]))

        dataset_frame = pd.DataFrame({"text": lines_text, "label": lines_label})
        return dataset_frame
