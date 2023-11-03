from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle

import torchvision
from corruptions import *


import random



class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        if self.params.agent == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                  self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def forward(self, x):
        return self.model.forward(x)

    def evaluate(self, test_loaders):

        acc_array = np.zeros(len(test_loaders))
        for task, test_loader in enumerate(test_loaders):
            acc = AverageMeter()
            print(task)
            for i, (batch_x, batch_y, indices_1) in enumerate(test_loader):
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                
                
                np.random.seed(0)
                random.seed(0)
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                
                
                #batch_x_ = batch_x[0]  # Taking the first image from the batch
                batch_x_pil = torchvision.transforms.functional.to_pil_image(torch.randn((3,32,32)).cpu())  # Convert to PIL image
                torch.tensor(gaussian_noise(batch_x_pil).astype(float) / 255.0, dtype = batch_x.dtype).to("cuda").permute(2,0,1).reshape(batch_x.shape)
                
                
                np.random.seed(0)
                random.seed(0)
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                
                if task == 0 and i == 0:
                    print(batch_y, batch_x[0][0][0])
                self.model.eval()
                with torch.no_grad():
                    logits = self.model.forward(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

                    

                acc.update(correct_cnt, batch_y.size(0))
            acc_array[task] = acc.avg()
        print(acc_array)

        return acc_array
