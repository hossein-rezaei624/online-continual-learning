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
from models import resnet1
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random
import torchvision.transforms as transforms
import torchvision
import math

from torch.utils.data import Dataset
import pickle


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
    
    
    def train_learner(self, x_train, y_train, task_number):        
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)

        
        unique_classes = set()
        for _, labels, indices_1 in train_loader:
            unique_classes.update(labels.numpy())
        

        device = "cuda"
        Model_Carto = ResNet18(len(unique_classes))
        Model_Carto = Model_Carto.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(Model_Carto.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
        

        mapping = {value: index for index, value in enumerate(unique_classes)}
        
        # Training
        Carto = torch.zeros((6, len(y_train)))
        for epoch_ in range(6):
            print('\nEpoch: %d' % epoch_)
            Model_Carto.train()
            train_loss = 0
            correct = 0
            total = 0
            confidence_epoch = []
            for batch_idx, (inputs, targets, indices_1) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)                
                targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)
                
                optimizer_.zero_grad()
                outputs = Model_Carto(inputs)
                soft_ = self.soft_(outputs)
                confidence_batch = []
        
                for i in range(targets.shape[0]):
                  confidence_batch.append(soft_[i,targets[i]].item())
                        
                loss = criterion_(outputs, targets)
                loss.backward()
                optimizer_.step()
        
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
                conf_tensor = torch.tensor(confidence_batch)
                Carto[epoch_, indices_1] = conf_tensor
                
            print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total, " ,loss:", train_loss/(batch_idx+1))

            scheduler_.step()


        Confidence_mean = Carto.mean(dim=0)
        Variability = Carto.std(dim=0)
        
        plt.scatter(Variability, Confidence_mean, s = 2)
        
        plt.xlabel("Variability") 
        plt.ylabel("Confidence") 
        
        plt.savefig('scatter_plot.png')


        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()
        
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y, indices_1 = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
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

        counter__ = 0
        for i in range(self.buffer.buffer_label.shape[0]):
            if self.buffer.buffer_label[i].item() in unique_classes:
                counter__ +=1

        top_n = counter__

        # Find the indices that would sort the array
        sorted_indices_1 = np.argsort(Confidence_mean.numpy())
        sorted_indices_2 = np.argsort(Variability.numpy())
        
        #top_indices_1 = sorted_indices_1[:top_n] #hard to learn
        #top_indices_sorted = top_indices_1 #hard to learn
        
        #top_indices_1 = sorted_indices_1[-top_n:] #easy to learn
        #top_indices_sorted = top_indices_1[::-1] #easy to learn
        
        #top_indices_1 = sorted_indices_2[-top_n:] #ambigiuous
        #top_indices_sorted = top_indices_1[::-1] #ambiguous


        ##top_indices_sorted = sorted_indices_1 #hard to learn
        
        top_indices_sorted = sorted_indices_1[::-1] #easy to learn

        ##top_indices_sorted = sorted_indices_2[::-1] #ambigiuous

        
        subset_data = torch.utils.data.Subset(train_dataset, top_indices_sorted)
        trainloader_C = torch.utils.data.DataLoader(subset_data, batch_size=self.batch, shuffle=False, num_workers=0)

        images_list = []
        labels_list = []
        
        for images, labels, indices_1 in trainloader_C:  # Assuming train_loader is your DataLoader
            images_list.append(images)
            labels_list.append(labels)
        
        all_images = torch.cat(images_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)


        # Extract the first 10 images
        images = [all_images[i] for i in range(25)]
        labels = [all_labels[i] for i in range(25)]
        
        # Make a grid from these images
        grid = torchvision.utils.make_grid(images, nrow=5)  # 5 images per row
        
        torchvision.utils.save_image(grid, 'grid_image.png')
        

        # Extract the first 10 images
        images = [all_images[-i] for i in range(25)]
        labels = [all_labels[-i] for i in range(25)]
        
        # Make a grid from these images
        grid = torchvision.utils.make_grid(images, nrow=5)  # 5 images per row
        
        torchvision.utils.save_image(grid, 'grid_image1.png')

        '''num_per_class = top_n//len(unique_classes)
        counter_class = [0 for _ in range(len(unique_classes))]
        full = [math.ceil(top_n/len(unique_classes)) for _ in range(len(unique_classes))]

        images_list_ = []
        labels_list_ = []
        
        for i in range(all_labels.shape[0]):
            if counter_class[mapping[all_labels[i].item()]] < (num_per_class + 1):
                counter_class[mapping[all_labels[i].item()]] += 1
                labels_list_.append(all_labels[i])
                images_list_.append(all_images[i])
            if counter_class == full:
                print("yessssss")
                break

        print("counter_class", counter_class)
        print("full", full)
        all_images_ = torch.stack(images_list_)
        all_labels_ = torch.stack(labels_list_)
        #print("all_images_.shapeall_images_.shape",all_images_.shape)
        print("all_labels_.shapeeee",all_labels_.shape)'''

        
        counter = 0
        for i in range(self.buffer.buffer_label.shape[0]):
            if self.buffer.buffer_label[i].item() in unique_classes:
                self.buffer.buffer_label[i] = all_labels.to(device)[counter]
                self.buffer.buffer_img[i] = all_images.to(device)[counter]
                counter +=1

        print("counter", counter)

        self.after_train()
