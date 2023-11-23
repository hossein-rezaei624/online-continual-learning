import torch.nn as nn
from models.resnet import ResNet18
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import random
import torchvision.transforms as transforms
import torchvision
import math

from torch.utils.data import Dataset
import pickle

    
soft_ = nn.Softmax(dim=1)

def distribute_samples(probabilities, M):
    # Normalize the probabilities
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate the number of samples for each class
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    
    # Check if there's any discrepancy due to rounding and correct it
    discrepancy = M - sum(samples.values())
    
    for key in samples:
        if discrepancy == 0:
            break
        if discrepancy > 0:
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            samples[key] -= 1
            discrepancy += 1

    return samples

    
def distribute_excess(lst):
    # Calculate the total excess value
    total_excess = sum(val - 500 for val in lst if val > 500)

    # Number of elements that are not greater than 500
    recipients = [i for i, val in enumerate(lst) if val < 500]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    lst = [val if val <= 500 else 500 for val in lst]
    
    # Distribute the average share
    for idx in recipients:
        lst[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        lst[idx] += 1
    
    # Cap values greater than 500
    for i, val in enumerate(lst):
        if val > 500:
            return distribute_excess(lst)
            break

    return lst


def CASP_update(train_loader, train_dataset, Epoch, x_train, y_train, buffer):
        
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
    reverse_mapping = {index: value for value, index in mapping.items()}


    # Initializing the dictionaries        
    confidence_by_class = {class_id: {epoch: [] for epoch in range(Epoch)} for class_id, __ in enumerate(unique_classes)}

    
    # Training
    Carto = torch.zeros((Epoch, len(y_train)))
    for epoch_ in range(Epoch):
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
            soft_ = soft_(outputs)
            confidence_batch = []
    
            # Accumulate confidences and counts
            for i in range(targets.shape[0]):
                confidence_batch.append(soft_[i,targets[i]].item())
                
                # Update the dictionary with the confidence score for the current class for the current epoch
                confidence_by_class[targets[i].item()][epoch_].append(soft_[i, targets[i]].item())

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

    mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in confidences} for class_id, confidences in confidence_by_class.items()}
    std_of_means_by_class = {class_id: torch.std(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(Epoch)])) for class_id, __ in enumerate(unique_classes)}
    

    Confidence_mean = Carto.mean(dim=0)
    Variability = Carto.std(dim=0)
    

    list_of_indices = []
    counter__ = 0
    for i in range(buffer.buffer_label.shape[0]):
        if buffer.buffer_label[i].item() in unique_classes:
            counter__ +=1
            list_of_indices.append(i)

    top_n = counter__

    # Find the indices that would sort the array
    sorted_indices_1 = np.argsort(Confidence_mean.numpy())
    sorted_indices_2 = np.argsort(Variability.numpy())
    


    ##top_indices_sorted = sorted_indices_1 #hard
    
    ##top_indices_sorted = sorted_indices_1[::-1] #simple

    top_indices_sorted = sorted_indices_2[::-1] #challenging

    
    subset_data = torch.utils.data.Subset(train_dataset, top_indices_sorted)
    trainloader_C = torch.utils.data.DataLoader(subset_data, batch_size=10, shuffle=False, num_workers=0)

    images_list = []
    labels_list = []
    
    for images, labels, indices_1 in trainloader_C:  # Assuming train_loader is your DataLoader
        images_list.append(images)
        labels_list.append(labels)
    
    all_images = torch.cat(images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)


    updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}
    
    dist = distribute_samples(updated_std_of_means_by_class, top_n)

    
    num_per_class = top_n//len(unique_classes)
    counter_class = [0 for _ in range(len(unique_classes))]

    if len(y_train) == top_n:
        condition = [num_per_class for _ in range(len(unique_classes))]
        diff = top_n - num_per_class*len(unique_classes)
        for o in range(diff):
            condition[o] += 1
    else:
        condition = [value for k, value in dist.items()]


    check_bound = len(y_train)/len(unique_classes)
    for i in range(len(condition)):
        if condition[i] > check_bound:
            condition = distribute_excess(condition)
            break

    
    images_list_ = []
    labels_list_ = []
    
    for i in range(all_labels.shape[0]):
        if counter_class[mapping[all_labels[i].item()]] < condition[mapping[all_labels[i].item()]]:
            counter_class[mapping[all_labels[i].item()]] += 1
            labels_list_.append(all_labels[i])
            images_list_.append(all_images[i])
        if counter_class == condition:
            break

    
    all_images_ = torch.stack(images_list_)
    all_labels_ = torch.stack(labels_list_)

    indices = torch.randperm(all_images_.size(0))
    shuffled_images = all_images_[indices]
    shuffled_labels = all_labels_[indices]
    
    buffer.buffer_label[list_of_indices] = shuffled_labels.to(device)
    buffer.buffer_img[list_of_indices] = shuffled_images.to(device)
    
