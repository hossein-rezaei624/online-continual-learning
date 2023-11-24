import torch
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


def distribute_samples(probabilities, M):
    # Normalize the probabilities
    # Sum up the total probability to use for normalization    
    total_probability = sum(probabilities.values())

    # Create a new dictionary with normalized probabilities
    # Each probability is divided by the total to ensure they sum up to 1
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate the number of samples for each class
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    
    # Check if there's any discrepancy due to rounding and correct it
    discrepancy = M - sum(samples.values())

    # Iterate over each class to adjust the number of samples
    for key in samples:
        # If there is no discrepancy, stop adjusting
        if discrepancy == 0:
            break
        # If we have fewer samples than M, add a sample to the current class
        if discrepancy > 0:
            samples[key] += 1
            discrepancy -= 1
        # If we have more samples than M, remove a sample from the current class if possible
        elif discrepancy < 0 and samples[key] > 0:
            samples[key] -= 1
            discrepancy += 1

    # Return the final distribution of samples
    return samples

    
def distribute_excess(condition, max_samples_per_class):
    # Calculate the total excess value
    total_excess = sum(val - max_samples_per_class for val in condition if val > max_samples_per_class)

    # Number of elements that are not greater than max_samples_per_class
    recipients = [i for i, val in enumerate(condition) if val < max_samples_per_class]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    condition = [val if val <= max_samples_per_class else max_samples_per_class for val in condition]
    
    # Distribute the average share
    for idx in recipients:
        condition[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        condition[idx] += 1
    
    # Cap values greater than max_samples_per_class
    for i, val in enumerate(condition):
        if val > max_samples_per_class:
            return distribute_excess(condition)
            break

    return condition


def CASP_update(train_loader, train_dataset, Epoch, x_train, y_train, buffer, params_name):
        
    # Identify unique classes in the dataset
    unique_classes = set()
    for _, labels, indices_1 in train_loader:
        unique_classes.update(labels.numpy())
    
    # Set the device and initialize the model
    device = "cuda"
    CASP_Model = ResNet18(len(unique_classes), params_name)
    CASP_Model = CASP_Model.to(device)
    
    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CASP_Model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Create mapping and reverse mapping for class indices
    class_index_mapping = {value: index for index, value in enumerate(unique_classes)}
    index_class_mapping = {index: value for value, index in class_index_mapping.items()}


    # Initialize dictionaries to store confidence scores by class        
    confidence_by_class = {class_id: {epoch: [] for epoch in range(Epoch)} for class_id, __ in enumerate(unique_classes)}

    
    # Training
    confidence_by_sample = torch.zeros((Epoch, len(y_train)))
    for epoch in range(Epoch):
        print('\nEpoch: %d' % epoch)
        CASP_Model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, indices_1) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)                
            targets = torch.tensor([class_index_mapping[val.item()] for val in targets]).to(device)
            
            # Zero gradients, forward pass, calculate loss, backward pass, and update weights
            optimizer_.zero_grad()
            outputs = Model_Carto(inputs)
            soft = nn.Softmax(dim=1)(outputs)
            batch_confidence_scores = []
    
            # Compute and store confidence scores
            for i in range(targets.shape[0]):
                batch_confidence_scores.append(soft[i,targets[i]].item())
                
                # Update the dictionary with the confidence score for the current class for the current epoch
                confidence_by_class[targets[i].item()][epoch].append(soft[i, targets[i]].item())

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            # Calculate training statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            # Store confidence scores for samples
            confidence_by_sample[epoch, indices_1] = torch.tensor(confidence_batch)
            
        print("Accuracy:", 100.*correct/total, ", and:", correct, "/", total, " ,loss:", train_loss/(batch_idx+1))

        scheduler.step()

    # Calculate mean and standard deviation of confidence scores
    mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in confidences} for class_id, confidences in confidence_by_class.items()}
    std_of_means_by_class = {class_id: torch.std(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(Epoch)])) for class_id, __ in enumerate(unique_classes)}
    
    # Compute overall confidence mean and variability
    Confidence_mean = confidence_by_sample.mean(dim=0)
    Variability = confidence_by_sample.std(dim=0)
    
    buffer_indices_list = []
    counter = 0
    for i in range(buffer.buffer_label.shape[0]):
        if buffer.buffer_label[i].item() in unique_classes:
            counter +=1
            buffer_indices_list.append(i)

    # Sorting indices based on confidence and variability
    sorted_indices_1 = np.argsort(Confidence_mean.numpy())
    sorted_indices_2 = np.argsort(Variability.numpy())

    ##top_indices_sorted = sorted_indices_1 #hard samples
    ##top_indices_sorted = sorted_indices_1[::-1] #simple samples
    top_indices_sorted = sorted_indices_2[::-1] #challenging samples

    # Create a new training subset
    challenging_subset = torch.utils.data.Subset(train_dataset, top_indices_sorted)
    challenging_loader = torch.utils.data.DataLoader(challenging_subset, batch_size=10, shuffle=False, num_workers=0)

    # Extract images and labels from the new training subset
    images_list = []
    labels_list = []
    for images, labels, indices_1 in challenging_loader:  # Assuming train_loader is your DataLoader
        images_list.append(images)
        labels_list.append(labels)
    
    all_images = torch.cat(images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Update the standard deviation of means by class
    updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}
    class_distribution = distribute_samples(updated_std_of_means_by_class, counter)

    num_per_class = top_n//len(unique_classes)
    counter_class = [0 for _ in range(len(unique_classes))]

    if len(y_train) == counter:
        condition = [num_per_class for _ in range(len(unique_classes))]
        diff = top_n - num_per_class*len(unique_classes)
        for o in range(diff):
            condition[o] += 1
    else:
        condition = [value for k, value in class_distribution.items()]

    max_samples_per_class = len(y_train)/len(unique_classes)
    for i in range(len(condition)):
        if condition[i] > max_samples_per_class:
            condition = distribute_excess(condition, max_samples_per_class)
            break
    
    images_list_ = []
    labels_list_ = []
    
    for i in range(all_labels.shape[0]):
        if counter_class[class_index_mapping[all_labels[i].item()]] < condition[class_index_mapping[all_labels[i].item()]]:
            counter_class[class_index_mapping[all_labels[i].item()]] += 1
            labels_list_.append(all_labels[i])
            images_list_.append(all_images[i])
        if counter_class == condition:
            break

    all_images_ = torch.stack(images_list_)
    all_labels_ = torch.stack(labels_list_)

    indices = torch.randperm(all_images_.size(0))
    shuffled_images = all_images_[indices]
    shuffled_labels = all_labels_[indices]
    
    buffer.buffer_label[buffer_indices_list] = shuffled_labels.to(device)
    buffer.buffer_img[buffer_indices_list] = shuffled_images.to(device)
    
