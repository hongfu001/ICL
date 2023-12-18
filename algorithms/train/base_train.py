from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

import torch


class Trainer:
    def __init__(self, args, net, device, train_loader):
        self.args = args
        
        self.net=net
        self.device = device
        self.train_loader = train_loader
        
        self.total_steps = len(self.train_loader)*args.epochs
        self.optimizer_model = AdamW(self.net.parameters(),
                                    lr = args.learning_rate
                                     )
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer_model,
                                                num_warmup_steps = args.warmup_steps,
                                                num_training_steps=self.total_steps)
        
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self, net, train_loader, device, epoch):
        self.net=net.to(device)
        self.net.train()
        loss_avg=0

        for _,input_ids, attention_mask, labels in tqdm(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            loss = self.train_batch(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels, 
                        epoch=epoch
                        )
            
            self.optimizer_model.zero_grad()
        
            loss.backward()
            self.optimizer_model.step()
            loss_avg = loss_avg + float(loss)
        self.scheduler.step()
        return loss_avg

    def train_batch(self, input_ids, attention_mask, labels, epoch):
        pass

    def test(self, net, test_loader):
        net.eval()
        net.to(self.device)
        loss_avg = 0
        correct = 0

        with torch.no_grad():
            for _,input_ids, attention_mask, labels in tqdm(test_loader):
                input_ids, attention_mask, labels=input_ids.to(self.device),attention_mask.to(self.device),labels.to(self.device)
                
                
                outputs = net(input_ids = input_ids,
                                attention_mask = attention_mask,
                                ).logits
                
                loss = self.loss_function(outputs, labels)
                loss_avg += float(loss.data)
                
                pred = outputs.data.max(1)[1]
                correct += pred.eq(labels.data).sum().item()

        return loss_avg / len(test_loader), correct / len(test_loader.dataset)