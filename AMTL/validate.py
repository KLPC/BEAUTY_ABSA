from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import pandas as pd
from loss import *

####################  1. VALIDATION UTILS    ####################


def to_device(device, batch):
    x_sent, x_sent_num, y1, y2,mask_mine,big_task,small_task, label = batch
    return ( x_sent.to(device), x_sent_num.to(device), y1.to(device), 
        y2.to(device), mask_mine.to(device), big_task.to(device), small_task.to(device), label.to(device) )

####################  2. VALIDATION (MAIN)    ####################

def get_impression( tg, task_num, small_task):
    imp = sum(( [[-1, 0.5, 1]] * np.exp( tg[task_num-1].detach().cpu()[torch.where( 
        small_task == task_num)[0]].numpy() 
                                 ) / np.expand_dims( 
        np.exp( tg[task_num-1].detach().cpu()[torch.where( small_task == task_num)[0]].numpy()).sum(axis = 1), 1) ).flatten())
    return imp

def validation(val_loader, model, config, writer, epoch_i):
    # DB
    task_names_lst = ['스킨케어:클렌징', '스킨케어:모이스쳐', '스킨케어:선케어', 
                      '메이크업1:페이스',
                      '메이크업2:립', 
                      '메이크업3:아이', '메이크업3:기타색조(치크)', 
                      '생활용품:바디', '생활용품:헤어', ]
    small_cat_indices = [[1, 10, 11],
                         [1, 6, 7],
                         [1, 8, 9],

                         [3, 4, 5],

                         [4, 5, 11, 12, 13, 14],

                         [4, 8, 13],
                         [4, 8, 13],

                         [2, 1, 0],
                         [2, 1, 0]]
    cols = ['용량','민감성','향기','커버력','지속력','피부톤',
             '보습감','향','사용감','발림성','세정력', # 10
             '촉촉함','유분기','발색감','제형','보습력']
    task_dict = dict(zip ( [ i for i in range(1,10)], [ list( np.array(cols)[i]) for i in small_cat_indices] ))

    lip_list = []
    

    # Member Function
    def append_results(val_over_lst, val_task_lst, out_task_lst, gt_task_lst, small_task, mask_mine, out_spec, label, task_specific_loss, overall_loss):
        for idx, (task_id, gt, val_loss_ts, val_loss_o) in enumerate(zip( small_task, label, 
                                                     np.array(task_specific_loss.detach().cpu().data), 
                                                      np.array(overall_loss.detach().cpu().data) )):
        #for task_id, msk, out, gt, val_loss_ts, val_loss_o in zip( small_task, y1_mask, out_spec, label, np.array(task_specific_loss.cpu().data), np.array(overall_loss.cpu().data) ):
            #print(task_id, loss_val)
            val_task_lst[task_id - 1].append(val_loss_ts)
            val_over_lst[task_id -1].append(val_loss_o)

            #outputs = torch.argmax( out, dim = 1).tolist()
            out_mine = out_spec[task_id-1][:, idx, :]
            out_mine = out_mine[ torch.tensor( [i in torch.where( mask_mine[idx].detach().cpu() == 1)[0].numpy() for i in small_cat_indices[task_id-1] ] ) ]
            
            #outputs = torch.argmax( out_spec[task_id -1][:, idx, :], dim = 1).tolist()
            outputs = torch.argmax( out_mine, dim = 1).tolist()
            labels = torch.argmax( gt [ mask_mine[idx] == 1], dim = 1).tolist()
            [ out_task_lst[task_id - 1][idx].append(item) for idx, item in enumerate(outputs)]
            [ gt_task_lst[task_id - 1][idx].append(item) for idx, item in enumerate(labels)]




        return val_over_lst, val_task_lst, out_task_lst, gt_task_lst

    # CONTAINER
    out_task1 = [[], [], []]
    out_task2 = [[], [], []]
    out_task3 = [[], [], []]
    out_task4 = [[], [], []]
    out_task5 = [[], [], [],  [], [], []]
    out_task6 = [[], [], []]
    out_task7 = [[], [], []]
    out_task8 = [[], [], []]
    out_task9 = [[], [], []]

    out_task_lst = [out_task1, out_task2, out_task3, out_task4, 
                 out_task5, out_task6, out_task7,out_task8, out_task9 ]

    gt_task1 = [[], [], []]
    gt_task2 = [[], [], []]
    gt_task3 = [[], [], []]
    gt_task4 = [[], [], []]
    gt_task5 = [[], [], [],  [], [], []]
    gt_task6 = [[], [], []]
    gt_task7 = [[], [], []]
    gt_task8 = [[], [], []]
    gt_task9 = [[], [], []]

    gt_task_lst = [gt_task1, gt_task2, gt_task3, gt_task4, 
                 gt_task5, gt_task6, gt_task7, gt_task8, gt_task9 ]

    val_task1,val_task2,val_task3,val_task4,val_task5 = [], [], [], [], []
    val_task6,val_task7,val_task8,val_task9 =[], [], [], []

    val_task_lst = [val_task1, val_task2, val_task3, val_task4, 
                 val_task5, val_task6, val_task7, val_task8, val_task9 ]

    val_over_lst = [ [], [], [], [], [], [], [], [], [] ]

    # LOSS
    overall_criterion = initialize_overall_loss(config)
    ts_criterion = initialize_ts_loss(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss_lst = []
    print('VALIDATION...')

    for idx, sample in enumerate(val_loader):
        # SAMPLE
        x_sent, x_sent_num, y1, y2, mask_mine, big_task, small_task, label = to_device(device, sample)

        # MODEL OUTPUT
        tg, ts, a1, a2, a3, a4, a5, a_g = model(x_sent, x_sent_num)            

        # CALCULATE LOSS
        overall_loss = calc_overall_loss(tg, y2, overall_criterion, big_task, val_flag = True)
        task_specific_loss = torch.stack( calc_task_specific_loss( ts, label, mask_mine, config, ts_criterion, small_task, small_cat_indices)) 
        total_loss = config['overall_wt']*overall_loss.mean() + (1-config['overall_wt']) * task_specific_loss.mean()
    
        total_loss_lst.append( total_loss.detach().cpu().data )

        val_over_lst, val_task_lst, out_task_lst, gt_task_lst = append_results(val_over_lst, 
                                                                val_task_lst, out_task_lst, gt_task_lst, 
                                                                small_task, mask_mine, ts, label, 
                                                                task_specific_loss, overall_loss)

    overall_val_loss = [np.mean(item) for item in val_over_lst]
    task_specific_val_loss = [np.mean(item) for item in val_task_lst]
    total_validation_loss = torch.mean( torch.stack(total_loss_lst).data ).detach().cpu().numpy()


    #print('='*80)
    #print('RESULT METRIC / LOSS COPUTATION =====>')
    idx = 0
    eps = 1e-20
    print('TOTAL VALIDATION LOSS: ', total_validation_loss)

    f1_lst = [0]
    spec_lst = [0]

    # printer
#     for i, (out_lst, gt_lst) in enumerate( zip(gt_task_lst, out_task_lst)):
#         print(f'\n[{i+1}] {task_names_lst[i]}')
#         print(f'(1) OVERALL VALIDATION LOSS: {overall_val_loss[i]:.4f} \
#                 (2) TASK SPECIFIC LOSS: {task_specific_val_loss[i]:.4f}')
#         writer.add_scalar(f'(Val) Task Specific/{task_names_lst[i]}', task_specific_val_loss[i], global_step = epoch_i)
#         writer.add_scalar(f'(Val) Overall /{task_names_lst[i]}', overall_val_loss[i], global_step = epoch_i)

#         for j, (o, gt) in enumerate( zip(out_lst, gt_lst)):
#             #print(j)
#             idx += 1
#             contingency_table = confusion_matrix(o, gt)
#             #print(contingency_table)
#             if contingency_table.shape == (3,3):  # 색조의 경우 binary
#                 TN = contingency_table[0,0] # -1 을 -1 로 예측한것
#                 FN1 = contingency_table[2,0] # 실제 긍정인데 부정이라고 예측한거
#                 FN2 = contingency_table[1,0] # 실제 중립인데 부정이라고 예측
#                 FN3 = contingency_table[2,1] # 실제 중립인데 긍정이라고 예측

#                 TP = contingency_table[1,1] + contingency_table[2,2] # 실제 중립/긍정을 정확히 예측
#                 FP = contingency_table[0, 1] + contingency_table[0, 2] # 실제 부정(-1) 인데 중립/긍정 이라고 예측
#                 FP2 = contingency_table[1, 2] # 실제로 중립(0)인데 긍정(1)이라고 예측함...

#                 specificity =  TN / (TN + FP + FP2 + eps)        # TNR
#                 precision = TP / (TP + FP + FP2 + eps)           # 1 - FDR
#                 sensitivity = TP / (TP + FN1 + FN2 + FN3 + eps)  # TPR
#             else:
#                 try: 
#                     tn, fp, fn, tp = contingency_table.ravel()
#                     specificity = tn / (tn + fp)
#                     precision = tp / (tp + fp)
#                     sensitivity = tp / (tp + fn)
#                 except:
#                     print('contingency exception handle....')
#                     specificity = 0.000000001
#                     precision = 0.000000001
#                     sensitivity = 0.000000001


# #            modified_f1 = 2 / ((1 / (specificity + eps)) + (1 / (precision + eps)))
#             print(f'{task_dict[i+1][j]}:specificity:{specificity:.3f}, precision:{precision:.3f},sensitivity:{sensitivity:.3f}')

    #         f1_lst.append(modified_f1)
    #         spec_lst.append(specificity)

    #         writer.add_scalar(f'Specificity/{task_names_lst[i]}', specificity, global_step = epoch_i)
    #         writer.add_scalar(f'Precision/{task_names_lst[i]}', precision, global_step = epoch_i)
    #         writer.add_scalar(f'Sensitivity/{task_names_lst[i]}', sensitivity, global_step = epoch_i)
    #         writer.add_scalar(f'F1/{task_names_lst[i]}', modified_f1, global_step = epoch_i)
    #         #writer.add_scalar('Valid/')

    return total_validation_loss, task_specific_val_loss, np.mean(f1_lst), np.mean(spec_lst)