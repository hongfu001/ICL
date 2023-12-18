import numpy as np
from torch.utils.data import DataLoader
from peft import PeftModel, get_peft_model, AdaLoraConfig, TaskType

import os
import torch
import time
import random
import argparse

from dataset.utils import build_dataset
from model.utils import build_model, build_tokenizer
from algorithms.train.utils import create_train_alg
from dataset.validation_dataset import validation_split


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def main(args):
    
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(args.seed)

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, num_classes = build_dataset(args, dataset_type='train', is_icl=args.is_icl, demo_type=args.alg_sample, sample_rate=args.sample_rate,random_state=args.seed)

    train_data, val_data = validation_split(train_dataset, val_share=0.1)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)
    
    tokenizer, model=build_tokenizer(args.model), build_model(args.model, num_classes)


    # Make save directory
    alg_train_obj = create_train_alg(args, model, device, train_loader)
    state = {k: v for k, v in args._get_kwargs()}
    

    #######################Test#######################
    state['test_loss'], state['test_accuracy'] = alg_train_obj.test(model, test_loader)
    print('Raw |Test Loss {0:.4f} | Test acc {1:.4f}|'.format(
        state['test_loss'],
        100. * state['test_accuracy']
    ))

    #######################Train_Test#######################
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()
        
        if args.alg_train=='lora':
            peft_config = AdaLoraConfig(
                                        task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                        r=8,
                                        lora_alpha=32, lora_dropout=0.1,
                                        target_modules=["q", "v"]
                                    )
            
            model = get_peft_model(model, peft_config)
        else:
            model = model
        
        state['train_loss'] = alg_train_obj.train(model, train_loader, device, epoch)
        
        model_tune = model.module if hasattr(model, 'module') else model
        
        model_tune.save_pretrained(os.path.join(args.save+"Epoch_"+str(epoch)))
        tokenizer.save_pretrained(os.path.join(args.save+"Epoch_"+str(epoch)))

        if args.alg_train=='lora':
            model_tune = PeftModel.from_pretrained(model, os.path.join(args.save+"Epoch_"+str(epoch)))
            model_tune = model_tune.merge_and_unload() #合并lora权重
        else:
            model_tune = model_tune

        state['val_loss'], state['val_accuracy']   = alg_train_obj.test(model_tune, val_loader)
        state['test_loss'], state['test_accuracy'] = alg_train_obj.test(model_tune, test_loader)
        
        print('Tune | Epoch {0:3d} | Time {1:5f} | Train Loss {2:.4f} | Val Loss {3:.4f} | | Val acc {4:.4f} | Test Loss {5:.4f} | Test acc {6:.4f}'.format(
            epoch + 1,
            time.time() - begin_epoch,
            state['train_loss'],
            state['val_loss'],
            state['val_accuracy'],
            state['test_loss'],
            100. * state['test_accuracy']
            ))


    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='0 = CPU.')

    parser.add_argument('--model', '-m', type=str, default='flan-t5', help='Choose architecture.')
    
    parser.add_argument('--is_icl',  type=bool, default=True, help='is ICL.')
    parser.add_argument('--alg_sample', '-alg_sample', type=str, default='random', help='Choose dataset architecture.')
    parser.add_argument('--sample_rate',  type=float, default=0.5, help='sample_rate')
    parser.add_argument('--num_demo',  type=int, default=4)
    parser.add_argument('--input',  type=str, default='text')
    parser.add_argument('--output', type=str, default='label')

    # Optimization options
    parser.add_argument('--alg_train', '-alg_train', type=str, default='lora', help='Choose train architecture.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.00001, help='The initial learning rate.')
    parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup_steps.')

    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
    args = parser.parse_args()
    
    main(args)