import time
import numpy as np
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from experiment.metrics import compute_performance, single_run_avg_end_fgt
from experiment.tune_hyperparam import tune_hyper
from types import SimpleNamespace
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
import pandas as pd
import os
import pickle


import torch
from utils.setup_elements import transforms_match
from continuum.data_utils import dataset_transform

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import Dataset
import pickle

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torchvision.models as models_tsne




def apply_tsne(features, labels, random_image_indices, perplexity=30, learning_rate=200, n_iter=1000):
    # Standardize features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=0)
    reduced_features = tsne.fit_transform(standardized_features)

    # Visualization
    plt.figure(figsize=(4, 4))
    colors = plt.cm.get_cmap('tab10', 10)  # Get a colormap with 10 distinct colors
    for i in range(10):
    
        indices = [j for j, label in enumerate(labels) if label == i]
        
        # Split indices based on whether they are in random_image_indices
        special_indices = [index for index in indices if index in random_image_indices]
        normal_indices = [index for index in indices if index not in random_image_indices]
    
        # Plotting
        color = colors(i)  # Get the color for the current class
        plt.scatter(reduced_features[normal_indices, 0], reduced_features[normal_indices, 1], color=color, alpha=0.2, label=f'Class {i}', s=1)
        if special_indices:
            plt.scatter(reduced_features[special_indices, 0], reduced_features[special_indices, 1], color=color, marker='^', label=f'Class {i} special', s=30)
    
    
    ##plt.legend()
    plt.savefig("tsneCASPvar77")




def multiple_run(params, store=False, save_path=None):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if store:
        result_path = load_yaml('config/global.yml', key='path')['result']
        table_path = result_path + params.data
        print(table_path)
        os.makedirs(table_path, exist_ok=True)
        if not save_path:
            save_path = params.model_name + '_' + params.data_name + '.pkl'

    accuracy_list = []
    for run in range(params.num_runs):
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        model = setup_architecture(params)
        model = maybe_cuda(model, params.cuda)
        opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
        agent = agents[params.agent](model, opt, params)

        
        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        if params.online:
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                print("-----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.before_train(x_train, y_train)
                agent.train_learner(x_train, y_train)


                if i == 0:
                    print("we are hereee 0")

                    x_train_tsne = x_train
                    y_train_tsne = y_train

                    train_dataset_tsne = dataset_transform(x_train, y_train, transform=transforms_match[agent.data])
                    train_loader_tsne = torch.utils.data.DataLoader(train_dataset_tsne, batch_size=10, shuffle=True, num_workers=0,
                                                   drop_last=True)
                                

                    # Load and modify the ResNet18 model for 10 classes
                    model_tsne = models_tsne.resnet18(pretrained=True)
                    num_ftrs_tsne = model_tsne.fc.in_features
                    model_tsne.fc = nn.Linear(num_ftrs_tsne, 10)  # 10 classes
                    model_tsne = model_tsne.to('cuda')
                    
                    criterion_CASP = nn.CrossEntropyLoss()
                    optimizer_CASP = optim.SGD(model_tsne.parameters(), lr=0.001,
                                          momentum=0.9, weight_decay=5e-4)
                    
                    
                    # Train the model
                    num_epochs_tsne = 7  # Adjust number of epochs as necessary
                    for epoch_tsne in range(num_epochs_tsne):
                        model_tsne.train()
                        running_loss_tsne = 0.0
                        correct_tsne = 0
                        total_tsne = 0
                        for inputs_tsne, labels_tsne, ___tsne in train_loader_tsne:
                            inputs_tsne, labels_tsne = inputs_tsne.to('cuda'), labels_tsne.to('cuda')
                            optimizer_CASP.zero_grad()
                            outputs_tsne = model_tsne(inputs_tsne)
                            loss_tsne = criterion_CASP(outputs_tsne, labels_tsne)
                            loss_tsne.backward()
                            optimizer_CASP.step()
                            running_loss_tsne += loss_tsne.item()
                    
                            # Calculate accuracy
                            __tsne, predicted_tsne = torch.max(outputs_tsne.data, 1)
                            total_tsne += labels_tsne.size(0)
                            correct_tsne += (predicted_tsne == labels_tsne).sum().item()
                    
                        epoch_loss_tsne = running_loss_tsne / len(train_loader_tsne)
                        epoch_accuracy_tsne = 100 * correct_tsne / total_tsne
                        print("\n")
                        print(f'Epoch {epoch_tsne+1}, Loss: {epoch_loss_tsne:.4f}, Accuracy: {epoch_accuracy_tsne:.2f}%')


                if i == 9:
                    print("now we are here 9")

                    
                    unique_classes_tsne = set()
                    for __tsne, labels_tsne, indices_1_tsne in train_loader_tsne:
                        unique_classes_tsne.update(labels_tsne.numpy())
                    
                    print("unique_classes_tsne", unique_classes_tsne)
                    list_of_indices_tsne = []
                    for i in range(agent.buffer.buffer_label.shape[0]):
                        if agent.buffer.buffer_label[i].item() in unique_classes_tsne:
                            list_of_indices_tsne.append(i)

                    print("len(list_of_indices_tsne)", len(list_of_indices_tsne))
                    train_dataset_CASP = dataset_transform(x_train_tsne, y_train_tsne, transform=transforms_match['cifar100'])
        
                    train_loader_CASP = torch.utils.data.DataLoader(train_dataset_CASP, batch_size=10, shuffle=False, num_workers=0,
                                                   drop_last=True)
                    
                    
                    random_image_indices_tsne = []
                    temp_again = agent.buffer.buffer_img[list_of_indices_tsne]
                    print("temp_again.shape", temp_again.shape)
                    # Iterate over the train_dataset
                    for idx_tsne, (data_11_tsne, target_tsne, ___tsne) in enumerate(train_dataset_CASP):
                        for random_img_tsne in temp_again:
                            # Compare data (image from train_dataset) with random_img
                            # The comparison logic depends on your data format
                            # For example, if they are numpy arrays or tensors you might do a direct comparison
                            if torch.equal(data_11_tsne.to('cuda'), random_img_tsne):
                                # If they match, store the index
                                random_image_indices_tsne.append(idx_tsne)
                                break  # Assuming each random image is unique
                    
                    
                    
                    
                    # Extract features for t-SNE
                    model_tsne.eval()
                    features_tsne = []
                    labels_tsne = []
                    with torch.no_grad():
                        for data__tsne, label_tsne, ___tsne in train_loader_CASP:
                            data__tsne, label_tsne = data__tsne.to('cuda'), label_tsne.to('cuda')
                            outputs_tsne = model_tsne(data__tsne)
                            features_tsne.extend(outputs_tsne.cpu().numpy())
                            labels_tsne.extend(label_tsne.cpu().numpy())
                    
                    # Convert features to a NumPy array
                    features_array_tsne = np.array(features_tsne)
                    labels_array_tsne = np.array(labels_tsne)
                    
                    # Apply t-SNE
                    apply_tsne(features_array_tsne, labels_array_tsne, random_image_indices_tsne, perplexity=50, learning_rate=300, n_iter=1000)
            
                    print("Now you can see the result...")


                

                

                agent.after_train()
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
            run_end = time.time()
            print(
                "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                               run_end - run_start))
            accuracy_list.append(np.array(tmp_acc))
        else:
            x_train_offline = []
            y_train_offline = []
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
            print('Training Start')
            x_train_offline = np.concatenate(x_train_offline, axis=0)
            y_train_offline = np.concatenate(y_train_offline, axis=0)
            print("----------run {} training-------------".format(run))
            print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
            agent.train_learner(x_train_offline, y_train_offline)
            acc_array = agent.evaluate(test_loaders)
            accuracy_list.append(acc_array)

    accuracy_array = np.array(accuracy_list)
    end = time.time()
    if store:
        result = {'time': end - start}
        result['acc_array'] = accuracy_array
        save_file = open(table_path + '/' + save_path, "wb")
        pickle.dump(result, save_file)
        save_file.close()
    if params.online:
        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    else:
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print("avg_end_acc {}".format(np.mean(accuracy_list)))




def multiple_run_tune(defaul_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(defaul_params.data, defaul_params.cl_type, defaul_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    #store table
    # set up storing table
    table_path = load_yaml('config/global.yml', key='path')['tables']
    metric_list = ['Avg_End_Acc'] + ['Avg_End_Fgt'] + ['Time'] + ["Batch" + str(i) for i in range(defaul_params.num_val, data_continuum.task_nums)]
    param_list = list(tune_params.keys()) + metric_list
    table_columns = ['Run'] + param_list
    table_path = table_path + defaul_params.data
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = defaul_params.model_name + '_' + defaul_params.data_name + '.csv'
    df = pd.DataFrame(columns=table_columns)
    # store list
    accuracy_list = []
    params_keep = []
    for run in range(defaul_params.num_runs):
        tmp_acc = []
        tune_data = []
        run_start = time.time()
        data_continuum.new_run()
        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), defaul_params)
        tune_test_loaders = test_loaders[:defaul_params.num_val]
        test_loaders = test_loaders[defaul_params.num_val:]
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < defaul_params.num_val:
                #collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == defaul_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, defaul_params, tune_params)
                    params_keep.append(best_params)
                    final_params = vars(defaul_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)

        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

        #store result
        result_dict = {'Run': run}
        result_dict.update(best_params)
        end_task_acc = tmp_acc[-1]
        for i in range(data_continuum.task_nums - defaul_params.num_val):
            result_dict["Batch" + str(i + defaul_params.num_val)] = end_task_acc[i]
        result_dict['Avg_End_Acc'] = np.mean(tmp_acc[-1])
        result_dict['Avg_End_Fgt'] = single_run_avg_end_fgt(np.array(tmp_acc))
        result_dict['Time'] = run_end - run_start
        df = df.append(result_dict, ignore_index=True)
        save_dataframe_csv(df, table_path, save_path)
    accuracy_list = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list)
    end = time.time()
    final_result = {'Run': 'Final Result'}
    final_result['Avg_End_Acc'] = avg_end_acc
    final_result['Avg_End_Fgt'] = avg_end_fgt
    final_result['Time'] = end - start
    df = df.append(final_result, ignore_index=True)
    save_dataframe_csv(df, table_path, save_path)
    print('----------- Total {} run: {}s -----------'.format(defaul_params.num_runs, end - start))
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))



def multiple_run_tune_separate(default_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(default_params.data, default_params.cl_type, default_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if default_params.num_val == -1:
        # offline tuning
        default_params.num_val = data_continuum.data_object.task_nums
    #store table
    # set up storing table
    result_path = load_yaml('config/global.yml', key='path')['result']
    table_path = result_path + default_params.data + '/' + default_params.cl_type
    for i in default_params.trick:
        if default_params.trick[i]:
            trick_name = i
            table_path = result_path + default_params.data + '/' + default_params.cl_type + '/' + trick_name
            break
    print(table_path)
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = default_params.model_name + '_' + default_params.data_name + '_' + str(default_params.seed) + '.pkl'
    # store list
    accuracy_list = []
    params_keep = []
    if isinstance(default_params.num_runs, int):
        run_list = range(default_params.num_runs)
    else:
        run_list = default_params.num_runs
    for run in run_list:
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        if default_params.train_val:
            single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        else:
            single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

    end = time.time()
    accuracy_array = np.array(accuracy_list)
    result = {'seed': default_params.seed}
    result['time'] = end - start
    result['acc_array'] = accuracy_array
    result['ram'] = check_ram_usage()
    result['best_params'] = params_keep
    save_file = open(table_path + '/' + save_path, "wb")
    pickle.dump(result, save_file)
    save_file.close()
    print('----------- Total {} run: {}s -----------'.format(default_params.num_runs, end - start))
    print('----------- Seed {} RAM: {}s -----------'.format(default_params.seed, result['ram']))

def single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    test_loaders = test_loaders_full[default_params.num_val:]

    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            else:
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0),
                      np.concatenate(labels_offline, axis=0))]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders)
        tmp_acc.append(acc_array)



def single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    break

        data_continuum.reset_run()
        # set up
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            print("----------run {} training batch {}-------------".format(run, i))
            print('size: {}, {}'.format(x_train.shape, y_train.shape))
            agent.train_learner(x_train, y_train)
            acc_array = agent.evaluate(test_loaders_full)
            tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            x_train_offline.append(x_train)
            y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0), labels_offline)]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders_full)
        tmp_acc.append(acc_array)

