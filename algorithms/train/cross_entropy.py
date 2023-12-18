import torch

from algorithms.train.base_train import Trainer

class CrossEntropy(Trainer):
    def train_batch(self, input_ids, attention_mask, labels, epoch):
        loss_function = torch.nn.CrossEntropyLoss()
        input_ids, inputs, targets = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        output = self.net(input_ids, inputs).logits
        loss = loss_function(output, targets)
        return loss