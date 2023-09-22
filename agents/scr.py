import torch #new strategy where we do fair...
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
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random
import torchvision.transforms as transforms
import torchvision
import math

from torch.utils.data import Dataset
import pickle


def load_mini_imagenet_cache(file_path):
    with open(file_path, 'rb') as f:
        data_cache = pickle.load(f)
    return data_cache

class MiniImageNetDataset(Dataset):
    def __init__(self, data_cache, transform=None):
        self.data = data_cache['image_data']
        
        # Create an integer mapping for class labels
        self.label_map = {label: idx for idx, label in enumerate(data_cache['class_dict'].keys())}
        
        # Convert class_dict to label format
        self.labels = [-1] * len(self.data)  # Initializing with a placeholder value
        for label, indices in data_cache['class_dict'].items():
            int_label = self.label_map[label]  # Getting the integer label
            for idx in indices:
                self.labels[idx] = int_label  # Assigning the integer label to the corresponding indices
                
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image) 
        return image, label


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
        #print("y_trainnnnnnn", y_train.shape, type(y_train), y_train)
        #print("x_trainnnnnnn", x_train.shape, type(x_train))
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        print("x_train[0]", x_train[0].reshape((3, 32, 32)), type(x_train[0]), x_train[0].shape)
        print("y_train[0]", y_train[0], type(y_train[0]), y_train[0].shape)
        a, b = train_dataset[0]
        print("image of data 0 is:", a, type(a), a.shape)
        print("label of data 0 is:", b, type(b), b.shape)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=False, num_workers=0,
                                       drop_last=True)


##        sets = [
##            [26, 86, 2, 55, 75, 93, 16, 73, 54, 95],
##            [53, 92, 78, 13, 7, 30, 22, 24, 33, 8],
##            [43, 62, 3, 71, 45, 48, 6, 99, 82, 76],
##            [60, 80, 90, 68, 51, 27, 18, 56, 63, 74],
##            [1, 61, 42, 41, 4, 15, 17, 40, 38, 5],
##            [91, 59, 0, 34, 28, 50, 11, 35, 23, 52],
##            [10, 31, 66, 57, 79, 85, 32, 84, 14, 89],
##            [19, 29, 49, 97, 98, 69, 20, 94, 72, 77],
##            [25, 37, 81, 46, 39, 65, 58, 12, 88, 70],
##            [87, 36, 21, 83, 9, 96, 67, 64, 47, 44]
##        ]

##        sets = [
##            [2, 8],
##            [4, 9],
##            [1, 6],
##            [7, 3],
##            [0, 5]
##        ]

##        sets = [
##            [49, 45, 14, 76, 73, 36, 21, 24, 84, 13],
##            [15, 82, 59, 35, 32, 55, 97, 41, 43, 77],
##            [54, 11, 42, 87, 95, 80, 33, 93, 83, 5],
##            [86, 48, 47, 67, 79, 65, 57, 12, 38, 60],
##            [72, 17, 26, 56, 25, 94, 51, 37, 50, 3],
##            [9, 31, 22, 4, 40, 75, 23, 78, 53, 16],
##            [62, 66, 85, 39, 71, 52, 2, 6, 90, 91],
##            [74, 27, 8, 88, 98, 70, 44, 20, 58, 61],
##            [1, 96, 28, 29, 34, 81, 19, 18, 69, 89],
##            [46, 10, 0, 7, 68, 99, 92, 30, 63, 64]
##        ]


##        transform_train = transforms.Compose([transforms.ToTensor(),])
##        train_cache = load_mini_imagenet_cache('datasets/mini_imagenet/mini-imagenet-cache-train.pkl')
##        trainset = MiniImageNetDataset(train_cache, transform=transform_train)
##        subset_indices_train = [idx for idx, (_, target) in enumerate(trainset) if target in sets[task_number]]
##        subset_loader_train = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, subset_indices_train),
##                                                          batch_size=10, shuffle=False, num_workers=0, drop_last=True)


        transform_train = transforms.Compose([transforms.ToTensor(),])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        a_ , b_ = trainset[0]
        print("from dataset, the image 0 is:", a_, type(a_), a_.shape)
        print("from dataset, the label 0 is:", b_, type(b_), b_.shape)
        ##subset_indices_train = [idx for idx, (_, target) in enumerate(trainset) if target in sets[task_number]]
        ##subset_loader_train = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, subset_indices_train),
        ##                                                  batch_size=10, shuffle=False, num_workers=0, drop_last=True)
        
        ##print("sets[task_number]", sets[task_number])
        
        unique_classes = set()
        
        count_ = np.sum(self.buffer.buffer_label.cpu().numpy() == 0)
        # Assuming each batch's labels are in the second element
        
        for _, labels in train_loader:
            unique_classes.update(labels.numpy())
        
        '''if count_ != self.buffer.buffer_label.shape[0]:
            unique_classes.update(self.buffer.buffer_label.cpu().numpy())
            #print("self.buffer.buffer_img.cpu().numpy().shape", self.buffer.buffer_img.cpu().numpy().shape)
            #print("self.buffer.buffer_img.permute(0, 3, 1, 2).cpu().numpy().shape", self.buffer.buffer_img.permute(0, 3, 1, 2).cpu().numpy().shape)
            train_dataset_buffer = dataset_transform(self.buffer.buffer_img.permute(0, 2, 3, 1).cpu().numpy(), 
                                                     self.buffer.buffer_label.cpu().numpy(), 
                                                     transform=transforms_match[self.data])
            #train_loader_buffer = data.DataLoader(train_dataset_buffer, batch_size=self.batch, shuffle=False, num_workers=0, drop_last=True)
        
            # Merge the two datasets
            merged_dataset = ConcatDataset([train_dataset, train_dataset_buffer])
            
            # Create a DataLoader for the merged dataset
            merged_loader = data.DataLoader(merged_dataset, batch_size=self.batch, shuffle=False, num_workers=0, drop_last=True)

        else:
            merged_dataset = train_dataset
            merged_loader = train_loader

        #print(f"Number of unique classes: {len(unique_classes)}", unique_classes)'''

        device = "cuda"
        Model_Carto = ResNet18(len(unique_classes))
        Model_Carto = Model_Carto.to(device)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.SGD(Model_Carto.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=200)
        
        #mapping = {26:0, 86:1, 2:2, 55:3, 75:4, 93:5, 16:6, 73:7, 54:8, 95:9,
         #         53:10, 92:11, 78:12, 13:13, 7:14, 30:15, 22:16, 24:17, 33:18, 8:19,
          #        43:20, 62:21, 3:22, 71:23, 45:24, 48:25, 6:26, 99:27, 82:28, 76:29,
           #       60:30, 80:31, 90:32, 68:33, 51:34, 27:35, 18:36, 56:37, 63:38, 74:39,
            #      1:40, 61:41, 42:42, 41:43, 4:44, 15:45, 17:46, 40:47, 38:48, 5:49,
             #     91:50, 59:51, 0:52, 34:53, 28:54, 50:55, 11:56, 35:57, 23:58, 52:59,
              #    10:60, 31:61, 66:62, 57:63, 79:64, 85:65, 32:66, 84:67, 14:68, 89:69,
               #   19:70, 29:71, 49:72, 97:73, 98:74, 69:75, 20:76, 94:77, 72:78, 77:79,
                #  25:80, 37:81, 81:82, 46:83, 39:84, 65:85, 58:86, 12:87, 88:88, 70:89,
                 # 87:90, 36:91, 21:92, 83:93, 9:94, 96:95, 67:96, 64:97, 47:98, 44:99}

        mapping = {value: index for index, value in enumerate(unique_classes)}
        #print(mapping)
        
        # Training
        Carto = []
        for epoch_ in range(6):
            print('\nEpoch: %d' % epoch_)
            Model_Carto.train()
            train_loss = 0
            correct = 0
            total = 0
            confidence_epoch = []
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                #print("targets", targets)
                
                
                targets = torch.tensor([mapping[val.item()] for val in targets]).to(device)
                #print("targets", targets)
                
                optimizer_.zero_grad()
                #print("inputs.shapeeeeeeeeeee", inputs.shape)
                outputs = Model_Carto(inputs)
                #print("outputs.shape", outputs.shape)
                soft_ = self.soft_(outputs)
                #print("soft_", soft_)
                confidence_batch = []
                #print("outputs", outputs)
        
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
        

            print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total, " ,loss:", train_loss/(batch_idx+1))
            conf_tensor = torch.tensor(confidence_epoch)
            conf_tensor = conf_tensor.reshape(conf_tensor.shape[0]*conf_tensor.shape[1])
            conf_tensor = conf_tensor[:total]
            #print(conf_tensor.shape)
            
            Carto.append(conf_tensor.numpy())
            scheduler_.step()

        Carto_tensor = torch.tensor(np.array(Carto))
        #print(Carto_tensor.shape)
        Confidence_mean = Carto_tensor.mean(dim=0)
        Variability = Carto_tensor.std(dim = 0)
        #print("Confidence_mean.shape", Confidence_mean.shape)
        #print("Variability.shape", Variability.shape)
        
        plt.scatter(Variability, Confidence_mean, s = 2)
        
        
        # Add Axes Labels
        
        plt.xlabel("Variability") 
        plt.ylabel("Confidence") 
        
        # Display
        
        plt.savefig('scatter_plot.png')



        #print("task_numberrrrrrrrrr", task_number)


##        if task_number > 0:
##    
##            space = self.params.mem_size
##            pointer = 0  # This will keep track of where to insert in M
##            
##            for j in range(task_number+1):  
##                portion = space // (task_number + 1 - j)  # Use integer division for portion size
##                
##                # Fill the buffer
##                for k in range(portion):
##                    if task_number != j:
##                        self.buffer.buffer_img[pointer] = self.buffer.buffer_img[j*self.params.mem_size//task_number + k]
##                        self.buffer.buffer_label[pointer] = self.buffer.buffer_label[j*self.params.mem_size//task_number + k]
##                        pointer += 1
##                    else:
##                        self.buffer.buffer_img[pointer] = all_images.to(device)[k]
##                        self.buffer.buffer_label[pointer] = all_labels.to(device)[k]
##                        pointer += 1
##                    
##                space -= portion

        
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()
        
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
                    #print("self.buffer.buffer_label", self.buffer.buffer_label)
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    #mem_x, mem_y = self.buffer.retrieve()
                    #print("in the loop mem_y.shape:", mem_y.shape)
                    #print("mem_x.shape", mem_x.shape)
                    #print("mem_y", mem_y)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        #print("mem_x.shape", mem_x.shape)
                        #print("batch_x.shape", batch_x.shape)
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


        #print("self.buffer.buffer_img", self.buffer.buffer_img.shape, type(self.buffer.buffer_img))
        #print("self.buffer.buffer_label", self.buffer.buffer_label.shape, type(self.buffer.buffer_label), self.buffer.buffer_label)

##        if count_ == self.buffer.buffer_label.shape[0]:
##            self.buffer.buffer_img = all_images.to(device)
##            self.buffer.buffer_label = all_labels.to(device)

        #print("self.buffer.buffer_img", self.buffer.buffer_img.shape, type(self.buffer.buffer_img))
        #print("self.buffer.buffer_label", self.buffer.buffer_label.shape, type(self.buffer.buffer_label), self.buffer.buffer_label)


        counter__ = 0
        for i in range(self.buffer.buffer_label.shape[0]):
            if self.buffer.buffer_label[i].item() in unique_classes:
                counter__ +=1


        # Number of top values you're interested in
        #top_n = (self.params.mem_size//(task_number+1)) + 1
        top_n = counter__


        # Find the indices that would sort the array
        sorted_indices_1 = np.argsort(Confidence_mean.numpy())
        sorted_indices_2 = np.argsort(Variability.numpy())
        
        # Take the last 'top_n' indices (i.e., the top values)
        #top_indices_1 = sorted_indices_1[:(top_n - (int(0.33*top_n) + int(0.33*top_n)))] #hard to learn
        #top_indices_2 = sorted_indices_1[-int(0.33*top_n):] #easy to learn
        #top_indices_3 = sorted_indices_2[-int(0.33*top_n):] #ambigiuous
        
        #print("top_indicesssss", top_indices.shape, top_indices, type(top_indices))

        #top_indices_12 = np.concatenate((top_indices_2, top_indices_3))
        #top_indices_123 = np.concatenate((top_indices_12, top_indices_1))
        
        #top_indices_sorted = top_indices_123

        
        
        # Take the last 'top_n' indices (i.e., the top values)
        #top_indices_1 = sorted_indices_1[:top_n]
        
        #top_indices_sorted = top_indices_1[::-1]
        #top_indices_sorted = top_indices_1


        top_indices_sorted = sorted_indices_1
        #top_indices_sorted = sorted_indices_2[::-1]

        
        #print("top_indices_sorted", top_indices_sorted, top_indices_sorted.shape)
        print("top_indices_sorted.shape", top_indices_sorted.shape)

        
        subset_data = torch.utils.data.Subset(train_dataset, top_indices_sorted)
        #print("subset_dataaaaaaaa", subset_data)
        trainloader_C = torch.utils.data.DataLoader(subset_data, batch_size=self.batch, shuffle=False, num_workers=0)

        images_list = []
        labels_list = []
        
        for images, labels in trainloader_C:  # Assuming train_loader is your DataLoader
            images_list.append(images)
            labels_list.append(labels)
        
        all_images = torch.cat(images_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        #print(all_images.shape)  # This should print something like torch.Size([50000, 3, 32, 32]) depending on your DataLoader's batch size
        #print(all_labels.shape)  # This should print torch.Size([50000])
        #print("all_labelsall_labels", all_labels)


        num_per_class = top_n//len(unique_classes)
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
        print("all_labels_.shapeeee",all_labels_.shape)

        
        print("unique_classes", unique_classes)
        counter = 0
        for i in range(self.buffer.buffer_label.shape[0]):
            if self.buffer.buffer_label[i].item() in unique_classes:
                self.buffer.buffer_label[i] = all_labels_.to(device)[counter]
                self.buffer.buffer_img[i] = all_images_.to(device)[counter]
                counter +=1

        print("counter", counter)

        #print("number of list(unique_classes)[0]", np.sum(self.buffer.buffer_label.cpu().numpy() == 2))
        
        self.after_train()

