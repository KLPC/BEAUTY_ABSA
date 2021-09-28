config = {}

# PATH
#config['data_path'] =  './' #'/content/drive/Shareddrives/AmorePacific2021/5.newcode/dataset'  #'../dataset'
#config['data_path'] = '/content/drive/Shareddrives/AmorePacific2021/5.newcode/MTL_ABSA_new/dataset'
config['data_path'] = './'
config['embed_path'] = './'

# MODE
config['embed'] = 'bert'
config['save_mode'] = True
config['check_label_cnts'] = False
config['grad_clip'] = True
config['gru_mode'] = True

# RECORD CONFIG
config['prefix'] = 'test'

#######################################################
# FIXED (DONT CHANGE)

## NETWORK CONFIG
config['doc_dim_nobert'] = 128
config['doc_dim_bert'] = 768
config['n_task1'] = 3     
config['n_task2'] = 3
config['n_task3'] = 3
config['n_task4'] = 3
config['n_task5'] = 6
config['n_task6'] = 3
config['n_task7'] = 3
config['n_task8'] = 3
config['n_task9'] = 3
config['n_class_o'] = 3    ## Rating (0~3)을 그냥 1로 둠
config['n_class_t'] = 3
config['sent_att_dim'] = 256
config['aspect_att_dim'] = 256

config['sent_gru_h_dim_nobert'] = 128
config['sent_gru_h_dim_bert'] = 768

# LOSS CONFIG
config['overall_wt'] = 0.5
config['clip_size'] = 1.0
config['ts_my_wt'] = 0.999

# PRINT CONFIG
config['step_size'] = 100

#######################################################
# VARIABLE (HYPERPARAMETER)

# NETWORK CONFIG
config['drop_1'] = 0.2
config['drop_2'] = 0.1
config['drop_3'] = 0.1
config['drop_4'] = 0.1
config['shared_dim1'] = 128
config['shared_dim2'] = 64
config['shared_dim3'] = 32
config['task_overall'] = 64
config['task1'] = 32
config['task2'] = 32
config['task3'] = 32
config['task4'] = 32
config['task5'] = 32
config['task6'] = 32
config['task7'] = 32
config['task8'] = 32
config['task9'] = 32
config['task_s'] = 16

# TRAIN CONFIG
config['lr'] = 1e-5
config['epochs'] = 40

# GRU
config['sent_gru_n_layers'] = 2
config['sent_gru_drop'] = 0.2

# LOSS CONFIG
config['focal_mode'] = True
config['ts_loss_mode'] = 'CE'
config['focal_alpha'] = 0.5
config['focal_gamma'] = 1.5

# DATA CONFIG
config['tr_ratio'] = 0.79999
config['val_ratio'] = 0.2
config['batch_size'] = 64




