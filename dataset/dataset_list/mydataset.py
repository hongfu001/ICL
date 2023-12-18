import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from dataset.dataset_list.tool import dataset_generation
from model.utils import build_Tokenizer
from sklearn import preprocessing

from transformers import  AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/dataset/home/llm_dev/gaohongfu/flan')


class MyDataSet:

    def __init__(self, args, dataset_type="train", is_icl=True, demo_type='random', sample_rate=0.1, random_state=10):
        global sample_test_dataset
        self.args = args
        self.dataset_type = dataset_type
        self.random_state = random_state
        self.sample_rate = sample_rate
        self.demo_type = demo_type
        self.is_icl = is_icl
        self.tokenizer = build_Tokenizer(args.model)
        self.le = preprocessing.LabelEncoder()

        self.train_list = ['climate']
        self.test_list = ['imdb']

        if dataset_type == "train":

            concat_train_dataset = pd.DataFrame()
            for dataset_one in self.train_list:
                train_dataset = dataset_generation(dataset_one)
                sample_train_dataset = dataset_generation(dataset_one).sample(frac=self.sample_rate,
                                                                              random_state=self.random_state, axis=0)
                if self.is_icl:
                    if demo_type == "random":
                        import sys
                        sys.path.append('..')
                        from algorithms.sampling.utils import create_sample_alg

                        alg_sample_obj = create_sample_alg(self.args, 'random')
                        dataset_one = alg_sample_obj.sample_demo(train_dataset, sample_train_dataset)
                        concat_train_dataset = pd.concat([concat_train_dataset, dataset_one])

                else:
                    concat_train_dataset = pd.concat([concat_train_dataset, sample_train_dataset])

            text = concat_train_dataset['text'].values
            label = concat_train_dataset['label'].values
            
            attention_mask = tokenizer(text.tolist(), padding="longest", max_length=1024, return_tensors="pt")
            self.index = attention_mask.input_ids
            self.train_text = attention_mask.attention_mask
            self.train_label = tokenizer(label.tolist(), padding="longest", max_length=1024, return_tensors="pt")
            


        else:
            import sys
            sys.path.append('..')
            from algorithms.sampling.utils import create_sample_alg

            concat_test_dataset = pd.DataFrame()
            if self.is_icl:
                for dataset_one in self.test_list:
                    test_dataset = dataset_generation(dataset_one)
                    sample_test_dataset = dataset_generation(dataset_one).sample(frac=self.sample_rate,
                                                                                 random_state=self.random_state, axis=0)

                    alg_sample_obj = create_sample_alg(self.args, 'random')
                    dataset_test_one = alg_sample_obj.sample_demo(test_dataset, sample_test_dataset)
                    concat_test_dataset = pd.concat([concat_test_dataset, dataset_test_one])


            else:
                for dataset_one in self.test_list:
                    concat_test_dataset = pd.concat([concat_test_dataset, sample_test_dataset])

            text = concat_test_dataset['text'].values
            label = concat_test_dataset['label'].values
            
            attention_mask = tokenizer(text.tolist(), padding="longest", max_length=1024, return_tensors="pt")
            self.index = attention_mask.input_ids
            self.test_text = attention_mask.attention_mask
            self.test_label = tokenizer(label.tolist(), padding="longest", max_length=1024, return_tensors="pt")
                


    def __getitem__(self, index):

        if self.dataset_type == "train":
            idx, text, target = self.index[index], self.train_text[index], self.train_label[index]
        else:
            idx, text, target =self.index[index], self.test_text[index], self.test_label[index]

        return idx, text, target

    def __len__(self):

        if self.dataset_type == "train":
            return len(self.train_text)

        # elif self.dataset_type == "valid":
        #     return len(self.valid_data)
        else:
            return len(self.test_text)
