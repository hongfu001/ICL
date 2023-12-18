from transformers import (
                        AutoTokenizer,
                        AutoModelForSequenceClassification
                          )
import torch

def build_model(model_type, num_classes):

    model_path={
        "gpt2":"gpt2",
        "flan-t5":"/data/home/llm_dev/gaohongfu/flan",
        "roberta":"roberta-base"
    }

    if model_type in model_path:
        model=AutoModelForSequenceClassification.from_pretrained(model_path[model_type], num_labels=num_classes)
        return model
    else:
        print("Error: model type")
        

def build_tokenizer(model_type):

    model_path={
        "gpt2":"gpt2",
        "flan-t5":"/data/home/llm_dev/gaohongfu/flan",
        "roberta":"roberta-base"
    }

    if model_type in model_path:
        Tokenizer=AutoTokenizer.from_pretrained(model_path[model_type])
        return Tokenizer
    else:
        print("Error: Tokenizer type")