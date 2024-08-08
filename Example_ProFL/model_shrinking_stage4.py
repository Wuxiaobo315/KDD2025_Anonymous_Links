import os
import copy
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update_res18 import LocalUpdate, LocalUpdate_shrinking,test_inference_init
from utils_cifar10 import get_dataset, average_weights, exp_details
import random
import math

from models_cifar10 import ResNet18_1,ResNet18_2,ResNet18_3,ResNet18_4,ResNet18_4_shrinking
import ast


def distri_device():
    entire_net = []
    first_stage = []
    second_stage = []
    third_stage = []
    forth_stage = []
    participant_ratio = []
    random_numbers = [random.randint(100, 900) for _ in range(100)]

    for i in range(len(random_numbers)):
        if random_numbers[i] >= 840:
            entire_net.append(i)
        if random_numbers[i] >= 478+28:
            first_stage.append(i)
        if random_numbers[i] >= 208+12:
            second_stage.append(i)
        if random_numbers[i] >= 108+4:
            third_stage.append(i)
        if random_numbers[i] >= 100:
            forth_stage.append(i)

    ratio1 = len(entire_net) / 100
    ratio2 = len(first_stage) / 100
    ratio3 = len(second_stage) / 100
    ratio4 = len(third_stage) / 100
    ratio5 = len(forth_stage) / 100
    participant_ratio = [ratio1, ratio2, ratio3, ratio4, ratio5]

    return participant_ratio, first_stage, second_stage, third_stage, forth_stage


if __name__ == '__main__':
    start_time = time.time()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    txtpath = 'resnet18_block4_shrinking.txt'
    f_acc = open(txtpath, 'a')
    f_acc.write(f'agrs = {args} \n')
    f_acc.flush()

    participant_ratio, first_stage, second_stage, third_stage, forth_stage = distri_device()
    f_acc.write(f'participant_ratio : {participant_ratio}  \n')
    f_acc.write(f'first stage user :{first_stage}\n')
    f_acc.write(f'second stage user :{second_stage}\n')
    f_acc.write(f'third stage user :{third_stage}\n')
    f_acc.write(f'forth stage user :{forth_stage}\n')
    f_acc.flush()

    train_dataset, test_dataset, user_groups = get_dataset(args)
    f_acc.write(f'load dataset finish and user_group is {user_groups} \n')
    f_acc.flush()


    record = [f_acc]
    global_model4 = ResNet18_4()

    device = 'cuda'
    global_model4.to(device)

    select_index1 = []
    select_index2 = []
    select_index3 = []
    select_index4 = []
    global_acc = []
    shrinking_global_acc = []
    all_signal = [] ####Determine according to block
    global_model = global_model4

    new_global_flag = 0
    for epoch in range(1,(args.epochs)+1):
        flash = 0

        if epoch>=0 and epoch<1000:
            flash=3 ### training fourth block

        idx_users = list(np.random.choice(forth_stage,20,replace=False))

        f_acc.write('-----------------------------------------------------------------\n')
        f_acc.write(f'global round {epoch} :{idx_users} \n')
        f_acc.flush()

        local_weights, local_losses = [], []
        after_scalar = {}
        for idx in idx_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss,update_local_model,accumulated_gradients,model_param = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch,idx = idx,flash=flash)
            local_weights.append(copy.deepcopy(w))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        if all_signal[0]==0:
            tmp1 = 0
            tmp_layer1 = []
            for name, child in enumerate(global_model.children()):
                if name == 4 and tmp1 == 0:
                    for param in child.parameters():
                        if param.requires_grad:
                            tmp_layer1.extend(torch.flatten(param))
                    tmp1 = 1 + tmp1
                    after_scalar[name] = tmp_layer1

            change_scalar = {}
            for key in after_scalar.keys():
                after_list = after_scalar[key]
                after_tensor = torch.tensor(after_list)
                change_scalar[key] = after_tensor
            for key in change_scalar.keys():

                total_number = len(change_scalar[key])
                if new_global_flag == 1:
                    if total_number < 10000:
                        select_index1=list(range(total_number))
                    elif total_number > 10000:
                        select_index_tmp = random.sample(range(total_number), 10000)
                        select_index1=list(select_index_tmp)

                if total_number > 10000:
                    record[0].write('[')
                    for index in select_index1:
                        number = change_scalar[key][index]
                        record[0].write(str(float(number)) + ',')
                    record[0].write('],')
                    record[0].flush()
                else:
                    record[0].write('[')
                    for number in change_scalar[key]:
                        record[0].write(str(float(number)) + ',')
                    record[0].write('],')
                    record[0].flush()

        if all_signal[0]==1 and all_signal[1]==0:
            tmp1 = 0
            tmp_layer1 = []
            for name, child in enumerate(global_model.children()):
                if name == 5 and tmp1 == 0:
                    for param in child.parameters():
                        if param.requires_grad:
                            tmp_layer1.extend(torch.flatten(param))
                    tmp1 = 1 + tmp1
                    after_scalar[name] = tmp_layer1

            change_scalar = {}
            for key in after_scalar.keys():
                after_list = after_scalar[key]
                after_tensor = torch.tensor(after_list)
                change_scalar[key] = after_tensor
            for key in change_scalar.keys():

                total_number = len(change_scalar[key])
                if new_global_flag == 1:
                    if total_number < 10000:
                        select_index2=list(range(total_number))
                    elif total_number > 10000:
                        select_index_tmp = random.sample(range(total_number), 10000)
                        select_index2=list(select_index_tmp)

                if total_number > 10000:
                    record[0].write('[')
                    for index in select_index2:
                        number = change_scalar[key][index]
                        record[0].write(str(float(number)) + ',')
                    record[0].write('],')
                    record[0].flush()
                else:
                    record[0].write('[')
                    for number in change_scalar[key]:
                        record[0].write(str(float(number)) + ',')
                    record[0].write('],')
                    record[0].flush()

        if all_signal[0]==1 and all_signal[1]==1 and all_signal[2]==0:
            tmp1 = 0
            tmp_layer1 = []
            for name, child in enumerate(global_model.children()):
                if name == 6 and tmp1 == 0:
                    for param in child.parameters():
                        if param.requires_grad:
                            tmp_layer1.extend(torch.flatten(param))
                    tmp1 = 1 + tmp1
                    after_scalar[name] = tmp_layer1

            change_scalar = {}
            for key in after_scalar.keys():
                after_list = after_scalar[key]
                after_tensor = torch.tensor(after_list)
                change_scalar[key] = after_tensor
            for key in change_scalar.keys():

                total_number = len(change_scalar[key])
                if new_global_flag == 1:
                    if total_number < 10000:
                        select_index3=list(range(total_number))
                    elif total_number > 10000:
                        select_index_tmp = random.sample(range(total_number), 10000)
                        select_index3=list(select_index_tmp)

                if total_number > 10000:
                    record[2].write('[')
                    for index in select_index3:
                        number = change_scalar[key][index]
                        record[2].write(str(float(number)) + ',')
                    record[2].write('],')
                    record[2].flush()
                else:
                    record[2].write('[')
                    for number in change_scalar[key]:
                        record[2].write(str(float(number)) + ',')
                    record[2].write('],')
                    record[2].flush()

        if all_signal[0] == 1 and all_signal[1] == 1 and all_signal[2] == 1:
            tmp1 = 0
            tmp_layer1 = []
            for name, child in enumerate(global_model.children()):
                if name == 7 and tmp1 == 0:
                    for param in child.parameters():
                        if param.requires_grad:
                            tmp_layer1.extend(torch.flatten(param))
                    tmp1 = 1 + tmp1
                    after_scalar[name] = tmp_layer1

            change_scalar = {}
            for key in after_scalar.keys():
                after_list = after_scalar[key]
                after_tensor = torch.tensor(after_list)
                change_scalar[key] = after_tensor
            for key in change_scalar.keys():

                total_number = len(change_scalar[key])
                if new_global_flag == 1:
                    if total_number < 10000:
                        select_index4 = list(range(total_number))
                    elif total_number > 10000:
                        select_index_tmp = random.sample(range(total_number), 10000)
                        select_index4 = list(select_index_tmp)

                if total_number > 10000:
                    record[0].write('[')
                    for index in select_index4:
                        number = change_scalar[key][index]
                        record[0].write(str(float(number)) + ',')
                    record[0].write('],')
                    record[0].flush()
                else:
                    record[0].write('[')
                    for number in change_scalar[key]:
                        record[0].write(str(float(number)) + ',')
                    record[0].write('],')
                    record[0].flush()

        if epoch>6  and epoch%50==0  :

            if all_signal[0]==0:
                with open('final_spmgfl_first_stage_layer1.txt', 'r') as file:
                    content = file.read()
                modified_content = content.strip() + ']'
                a = ast.literal_eval(modified_content)

                change1 = []
                tmp_length = len(a)
                import numpy as np

                for i in range(6, tmp_length):
                    tmp = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])) + (np.array(a[i - 1]) - np.array(a[i - 2])) + \
                                         (np.array(a[i - 2]) - np.array(a[i - 3])) + (np.array(a[i - 3]) - np.array(a[i - 4])) + \
                                         ((np.array(a[i - 4]) - np.array(a[i - 5]))) + (
                                         (np.array(a[i - 5]) - np.array(a[i - 6]))),
                                         ord=1)
                    norm = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])), ord=1) + np.linalg.norm(
                        (np.array(a[i - 1]) - np.array(a[i - 2])), ord=1) + \
                           np.linalg.norm((np.array(a[i - 2]) - np.array(a[i - 3])), ord=1) + np.linalg.norm(
                        (np.array(a[i - 3]) - np.array(a[i - 4])), ord=1) + \
                           np.linalg.norm((np.array(a[i - 4]) - np.array(a[i - 5])), ord=1) + np.linalg.norm(
                        (np.array(a[i - 5]) - np.array(a[i - 6])), ord=1)

                    tmp = (tmp) / (norm)
                    change1.append(tmp)
                f_acc.write(f'global_epoch{epoch}_change1={change1}\n')
                f_acc.flush()

            if all_signal[0]==1 and all_signal[1]==0:
                with open('second_cifar10_iid_pmgFL2_2.txt', 'r') as file:
                    content = file.read()
                modified_content = content.strip() + ']'
                a = ast.literal_eval(modified_content)

                change2 = []

                tmp_length = len(a)
                if tmp_length>6:
                    import numpy as np

                    for i in range(6, tmp_length):
                        tmp = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])) + (np.array(a[i - 1]) - np.array(a[i - 2])) + \
                                             (np.array(a[i - 2]) - np.array(a[i - 3])) + (
                                                         np.array(a[i - 3]) - np.array(a[i - 4])) + \
                                             ((np.array(a[i - 4]) - np.array(a[i - 5]))) + (
                                                 (np.array(a[i - 5]) - np.array(a[i - 6]))),
                                             ord=1)
                        norm = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 1]) - np.array(a[i - 2])), ord=1) + \
                               np.linalg.norm((np.array(a[i - 2]) - np.array(a[i - 3])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 3]) - np.array(a[i - 4])), ord=1) + \
                               np.linalg.norm((np.array(a[i - 4]) - np.array(a[i - 5])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 5]) - np.array(a[i - 6])), ord=1)

                        tmp = (tmp) / (norm)
                        change2.append(tmp)
                    f_acc.write(f'global_epoch{epoch}_change2={change2}\n')
                    f_acc.flush()

            if all_signal[0]==1 and all_signal[1]==1 and all_signal[2]==0:

                with open('cifar10_iid_pmgFL3.txt', 'r') as file:
                    content = file.read()

                modified_content = content.strip() + ']'
                a = ast.literal_eval(modified_content)

                change3 = []

                tmp_length = len(a)
                if tmp_length>6:
                    import numpy as np

                    for i in range(6, tmp_length):
                        tmp = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])) + (np.array(a[i - 1]) - np.array(a[i - 2])) + \
                                             (np.array(a[i - 2]) - np.array(a[i - 3])) + (
                                                         np.array(a[i - 3]) - np.array(a[i - 4])) + \
                                             ((np.array(a[i - 4]) - np.array(a[i - 5]))) + (
                                                 (np.array(a[i - 5]) - np.array(a[i - 6]))),
                                             ord=1)
                        norm = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 1]) - np.array(a[i - 2])), ord=1) + \
                               np.linalg.norm((np.array(a[i - 2]) - np.array(a[i - 3])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 3]) - np.array(a[i - 4])), ord=1) + \
                               np.linalg.norm((np.array(a[i - 4]) - np.array(a[i - 5])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 5]) - np.array(a[i - 6])), ord=1)

                        tmp = (tmp) / (norm)
                        change3.append(tmp)
                    f_acc.write(f'global_epoch{epoch}_change3={change3}\n')
                    f_acc.flush()

            if all_signal[0]==1 and all_signal[1]==1 and all_signal[2]==1:

                with open('acc_model_shrinking_stage4.txt', 'r') as file:
                    content = file.read()

                modified_content = content.strip() + ']'
                a = ast.literal_eval(modified_content)

                change4 = []

                tmp_length = len(a)
                if tmp_length>6:
                    import numpy as np

                    for i in range(6, tmp_length):
                        tmp = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])) + (np.array(a[i - 1]) - np.array(a[i - 2])) + \
                                             (np.array(a[i - 2]) - np.array(a[i - 3])) + (
                                                         np.array(a[i - 3]) - np.array(a[i - 4])) + \
                                             ((np.array(a[i - 4]) - np.array(a[i - 5]))) + (
                                                 (np.array(a[i - 5]) - np.array(a[i - 6]))),
                                             ord=1)
                        norm = np.linalg.norm((np.array(a[i]) - np.array(a[i - 1])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 1]) - np.array(a[i - 2])), ord=1) + \
                               np.linalg.norm((np.array(a[i - 2]) - np.array(a[i - 3])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 3]) - np.array(a[i - 4])), ord=1) + \
                               np.linalg.norm((np.array(a[i - 4]) - np.array(a[i - 5])), ord=1) + np.linalg.norm(
                            (np.array(a[i - 5]) - np.array(a[i - 6])), ord=1)

                        tmp = (tmp) / (norm)
                        change4.append(tmp)
                    f_acc.write(f'global_epoch{epoch}_change4={change4}\n')
                    f_acc.flush()


        if epoch %100==0:
            torch.save(global_model.state_dict(),f'resnet18_block4_{epoch}.pth')

        test_acc0, test_loss0 = test_inference_init(args, global_model,test_dataset)
        global_acc.append(test_acc0)
        print('global_acc =',global_acc)
        f_acc.write(f'global_acc = {global_acc} \n')
        f_acc.write('***************************\n')
        f_acc.flush()
