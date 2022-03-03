from config import *
from Impression import *
#from Variety import 
from data import *
from loss import * 
from validate import * 

import time
import datetime
import random
import transformers
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


####################  1. TRAINER UTILS    ####################

def get_model(config):
    model = MultiABSA(config)
    return model

def get_optimizer(model, config):
    print('Learning rate :', config['lr'])#wandb.config.learning_rate)
    optimizer = AdamW(model.parameters(), lr= config['lr'], eps=1e-8) # wandb.config.learning_rate, eps=1e-8)
    return optimizer

def get_scheduler(config, train_dataloader, optimizer):
    epochs = config['epochs']  #wandb.config.epochs
    print('Epochs =>', epochs)
    total_steps = len(train_dataloader) * epochs    ## [# of batches] * [# of epochs]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return scheduler

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def print_init_config(config):
    print('='*80)
    print('CONFIG...')
    print(config)
    print('='*80)

def to_device(device, batch):
    x_sent, x_sent_num, y1, y2,mask_mine,big_task,small_task, label = batch
    return ( x_sent.to(device), x_sent_num.to(device), y1.to(device), 
        y2.to(device), mask_mine.to(device), big_task.to(device), small_task.to(device), label.to(device) )

def today_timeinfo():
    from datetime import datetime
    today = str(datetime.now().today())
    return today[5:7] +today[8:10] + '_' + today[11:13] + today[14:16]

#################### 2 . TRAINER (MAIN)    ####################

def trainer(config, tr_loader, val_loader):
    print_init_config(config)
    with open('small_cat_indices.pkl', 'rb') as f:
        small_cat_indices = pickle.load(f)

    print('INITIALIZE TRAINING...')

    print('2. INITIALIZE MODEL...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize embed dim
    config['embed_dim'] = config['doc_dim_bert'] if config['embed'] == 'bert' else config['doc_dim_nobert']
    config['sent_att_dim'] = config['embed_dim']
    config['sent_gru_h_dim'] = config['sent_gru_h_dim_bert'] if config['embed'] == 'bert' else config['sent_gru_h_dim_nobert']

    print(device)
    model = get_model(config)
    model.to(device)
    
    print('3. INITIALIZE OPTIMIZER & SCHEDULER')
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(config, tr_loader, optimizer)
    
    # LOSS
    overall_criterion = initialize_overall_loss(config)
    ts_criterion = initialize_ts_loss(config)
    
    print('loss criterion overall: ', overall_criterion)
    print('loss criterion task specific: ', ts_criterion)
    
    # RANDOM SEED
    print('randomizing seed...')
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    training_stats = []
    total_t0 = time.time()
    #epochs = wandb.config.epochs
    
    train_losses, validation_losses = [], []

    # tensorboard
    writer = SummaryWriter('runs/' + config['prefix'])

    # model_directory
    model_path = './model_weight'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    weight_path = os.path.join(model_path, config['prefix']) 
    if not os.path.exists(weight_path):
        os.mkdir( os.path.join(weight_path ))

    for epoch_i in range(0, config['epochs']):
        
        ## Training
        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i+1, config['epochs']))
        print('Training ....')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        overall_loss_val = 0
        task_speicifc_loss_val = 0
        
        for idx, batch in enumerate(tr_loader):
            model.zero_grad()
            
            if idx % config['step_size'] == 0 and not idx == 0:
                elapsed = format_time(time.time() - t0)
                print('Data:',idx * config['batch_size'],'Elapsed:', elapsed)

            #x_sent, y1, y1_mask, y2, big_task, small_task, label = batch #to_device(batch, device)            
            x_sent, x_sent_num, y1, y2, mask_mine, big_task, small_task, label = to_device(device, batch)

            # MODEL OUTPUT
            tg, ts, a1, a2, a3, a4, a5, a_g = model(x_sent, x_sent_num)            
            
            # CALCULATE LOSS
            overall_loss = calc_overall_loss(tg, y2, overall_criterion, big_task)
            task_specific_loss = torch.stack( calc_task_specific_loss( ts, label, mask_mine, config, ts_criterion, small_task, small_cat_indices)) 
            total_loss = config['overall_wt']*overall_loss.mean() + (1-config['overall_wt']) * task_specific_loss.mean()
            
            #wandb.log({'train_batch_loss':loss.item()})
            total_loss.backward()

            overall_loss_val += overall_loss.detach().cpu().mean()
            task_speicifc_loss_val += task_specific_loss.detach().cpu().mean()
            total_train_loss += total_loss.detach().cpu().numpy()
            
            if config['grad_clip']:
                nn.utils.clip_grad_norm_(model.parameters(), config['clip_size'])
                
            optimizer.step()
            scheduler.step()

            
        avg_train_loss = total_train_loss / len(tr_loader)
        avg_overall_loss = overall_loss_val / len(tr_loader)
        avg_specific_loss = task_speicifc_loss_val / len(tr_loader)

        training_time = format_time(time.time() - t0)
        #wandb.log({'avg_train_loss':avg_train_loss})

        print('\nAverage training loss : {0:.4f}...'.format(avg_train_loss))
        print('Average training loss(overall) : {0:.4f}...'.format(avg_overall_loss))
        print('Average training loss(task) : {0:.4f}...'.format(avg_specific_loss))
        print('Training epoch took : {:}'.format(training_time))

        writer.add_scalar('Train/1.total', avg_train_loss, global_step = epoch_i)
        writer.add_scalar('Train/2.overall', avg_overall_loss, global_step = epoch_i)
        writer.add_scalar('Train/3.specific', avg_specific_loss, global_step = epoch_i)

        ## Validation
        print('Running Validation ....')
        t0 = time.time()
        model.eval()

        with torch.no_grad():
            total_validation_loss, task_specific_val_loss, modified_f1, specificity = validation(val_loader, model, config, writer, epoch_i)

        validation_time = format_time(time.time() - t0)
        writer.flush()
        #wandb.log({'val_accuracy':avg_val_acc, 'avg_val_loss':avg_val_loss})
        print('Validation took : {:}'.format(validation_time))

        training_stats.append(
            {
                'epoch' : epoch_i+1,
                'Train Loss' : avg_train_loss,
                'Train Loss(overall)' : avg_overall_loss, 
                'Train Loss(task)' : avg_specific_loss,
                'Valid Loss total' : total_validation_loss,
                'Valid Loss specific' : task_specific_val_loss,
                'F1 valid' : modified_f1,
                'Specificity' : specificity,
                'Training Time' : training_time,
                'Valid Time' : validation_time
            }
        )

        if config['save_mode'] == True:
            torch.save(model, weight_path + '/' + 'checkpoint' + str(epoch_i) + '.pt')


    print('Training complete !!')
    print('Total Training took {:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))

    return model, training_stats