from torch.utils.data import Dataset
from sklearn import preprocessing

import torch
import pandas as pd

from model.utils import build_tokenizer
from dataset.dataset_list.tool import dataset_generation
from algorithms.sampling.utils import create_sample_alg



class BaseDataSet(Dataset):

    def __init__(self, args, dataset_type="train", is_icl=True, demo_type='random', sample_rate=0.1, random_state=10):
        global sample_test_dataset
        self.args = args
        self.dataset_type = dataset_type
        self.random_state = random_state
        self.sample_rate = sample_rate
        self.demo_type = demo_type
        self.is_icl = is_icl
        self.tokenizer = build_tokenizer(args.model)
        
        self.pre = preprocessing.LabelEncoder()

        self.train_list = ["tweet_eval-stance_feminist", "ethos-national_origin", "tweet_eval-hate", "ag_news", "amazon_polarity", "hate_speech18", "poem_sentiment", "climate_fever", "medical_questions_pairs", "tweet_eval-stance_atheism", "superglue-cb", "dbpedia_14", "wiki_qa", "emo", "yelp_polarity", "ethos-religion", "financial_phrasebank", "tab_fact", "anli", "ethos-race"]
        self.test_list = ["imdb", "ethos-disability", "glue-wnli", "scitail", "trec-finegrained"]

        if dataset_type == "train":

            concat_train_dataset = pd.DataFrame()
            for dataset_one in self.train_list:
                train_dataset = dataset_generation(dataset_one)
                sample_train_dataset = dataset_generation(dataset_one).sample(frac=self.sample_rate,
                                                                              random_state=self.random_state, axis=0)
                if self.is_icl:
                    if demo_type == "random":
                        alg_sample_obj = create_sample_alg(self.args, 'random')
                        dataset_one = alg_sample_obj.sample_demo(train_dataset, sample_train_dataset)
                        concat_train_dataset = pd.concat([concat_train_dataset, dataset_one])

                else:
                    concat_train_dataset = pd.concat([concat_train_dataset, sample_train_dataset])

            text = concat_train_dataset['text']
            label = concat_train_dataset['label']
            self.num_classes = len(concat_train_dataset['label'].unique())
            
            train_encodings = self.tokenizer.batch_encode_plus(text.tolist(), padding=True, truncation=True, max_length=1024, return_tensors='pt')
            self.train_data =train_encodings['attention_mask']
            self.input_ids = train_encodings['input_ids']
            target = self.pre.fit_transform(label.tolist())
            self.train_target = torch.as_tensor(target)

        else:
            
            concat_test_dataset = pd.DataFrame()
            for dataset_one in self.test_list:
                test_dataset = dataset_generation(dataset_one)
                sample_test_dataset = dataset_generation(dataset_one).sample(frac=self.sample_rate,
                                                                                 random_state=self.random_state, axis=0)
                if self.is_icl:
                    alg_sample_obj = create_sample_alg(self.args, 'random')
                    dataset_test_one = alg_sample_obj.sample_demo(test_dataset, sample_test_dataset)
                    concat_test_dataset = pd.concat([concat_test_dataset, dataset_test_one])


                else:
                    concat_test_dataset = pd.concat([concat_test_dataset, sample_test_dataset])

            text = concat_test_dataset['text']
            label = concat_test_dataset['label']
            self.num_classes= len(concat_test_dataset['label'].unique())
            
            test_encodings = self.tokenizer.batch_encode_plus(text.tolist(), padding=True, truncation=True, max_length=1024, return_tensors='pt')
            self.test_data =test_encodings['attention_mask']
            self.input_ids = test_encodings['input_ids']
            
            target = self.pre.fit_transform(label.tolist())
            self.test_target = torch.as_tensor(target)       


    def __getitem__(self, index):

        if self.dataset_type == "train":
             input_ids,attention_mask, target = self.input_ids[index],self.train_data[index], self.train_target[index]
        else:
             input_ids,attention_mask, target = self.input_ids[index],self.test_data[index], self.test_target[index]

        return index, input_ids,attention_mask, target

    def __len__(self):

        if self.dataset_type == "train":
            return len(self.train_data)

        # elif self.dataset_type == "valid":
        #     return len(self.valid_data)
        else:
            return len(self.test_data)


    def get_num_classes(self):
        return self.num_classes
