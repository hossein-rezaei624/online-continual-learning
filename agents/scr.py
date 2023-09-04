import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn

from models.resnet import ResNet18
import numpy as np
import torch.optim as optim
import torch.nn as nn

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
        self.soft_ = nn.Softmax(dim=1)

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=False, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()
        
        unique_classes = set()
        
        count_ = np.sum(self.buffer.buffer_label.cpu().numpy() == 0)
        # Assuming each batch's labels are in the second element
        
        for _, labels in train_loader:
            unique_classes.update(labels.numpy())
        
        if count_ != self.buffer.buffer_label.shape[0]:
            unique_classes.update(self.buffer.buffer_label.cpu().numpy())
        print(f"Number of unique classes: {len(unique_classes)}", unique_classes)

        device = "cuda"
        Model_Carto = ResNet18(len(unique_classes))
        Model_Carto = Model_Carto.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(Model_Carto.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
        
        # Training
        for epoch_ in range(6):
            print('\nEpoch: %d' % epoch_)
            Model_Carto.train()
            train_loss = 0
            correct = 0
            total = 0
            confidence_epoch = []
            Carto = []
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_.zero_grad()
                outputs = Model_Carto(inputs)
                #print("outputs.shape", outputs.shape)
                soft_ = self.soft_(outputs)
                #print("soft_", soft_)
                confidence_batch = []
        
                for i in range(targets.shape[0]):
                  confidence_batch.append(soft_[i,targets[i]].item())
                if (targets.shape[0] != self.batch):
                  for j in range(self.batch - targets.shape[0]):
                    confidence_batch.append(0)
                confidence_epoch.append(confidence_batch)
                #print(len(confidence_epoch[0]))
        
                loss = criterion_(outputs, targets)
                loss.backward()
                optimizer_.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        

            print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total)
            conf_tensor = torch.tensor(confidence_epoch)
            conf_tensor = conf_tensor.reshape(conf_tensor.shape[0]*conf_tensor.shape[1])
            conf_tensor = conf_tensor[:(total-1)]
            #print(conf_tensor.shape)
            
            Carto.append(conf_tensor.numpy())
            scheduler_.step()

        Carto_tensor = torch.tensor(np.array(Carto))
        #print(Carto_tensor.shape)
        Confidence_mean = Carto_tensor.mean(dim=0)
        Variability = Carto_tensor.std(dim = 0)
        #print(Confidence_mean.shape)
        #print(Variability.shape)
        
        plt.scatter(Variability, Confidence_mean, s = 2)
        
        
        # Add Axes Labels
        
        plt.xlabel("Variability") 
        plt.ylabel("Confidence") 
        
        # Display
        
        plt.savefig('scatter_plot.png')

        

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for j in range(self.mem_iters):
                    #print("iiiiii", i)
                    #print("in the loop batch_y:", batch_y)
                    #print("buffer.current_index innn", self.buffer.current_index)
                    #print("self.buffer.buffer_label in:", self.buffer.buffer_label, "self.buffer.buffer_label.shape in:", self.buffer.buffer_label.shape)
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    #mem_x, mem_y = self.buffer.retrieve()
                    #print("in the loop mem_y.shape:", mem_y.shape)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_batch_aug = self.transform(combined_batch)
                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, combined_labels)
                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                                .format(i, losses.avg(), acc_batch.avg())
                        )


        
        self.after_train()
