import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
#import pickle5 as pickle
import pickle
import os

####################  1. PREPROCESS MODULE    ####################

def preprocess(data_path, model_type='w2v'):
  # ----------------- (1) Import Dataset ----------------- #
    data = pd.read_csv(os.path.join(data_path, 'amore_data_above8_rating.csv'))
    cols=list(data.columns[data.columns.str.contains('A')])+['category','rating', 'review_split']
    aspect_cols=list(data.columns[data.columns.str.contains('A')])
    data_embed = data[cols]

    # ----------------- (2) Import Embeddings ----------------- #
    assert model_type in ['w2v','glove','bert']
    embed_path = os.path.join(data_path, 'review_embed_above8_{}.pkl'.format(model_type))
    with open(embed_path, 'rb') as f:
        review_embed = pickle.load(f)
    data_embed['embed_sents']=review_embed

    # ----------------- (3) Add BIG Category Columns ----------------- #ㅁ
    cat_Bigcat={'cat1':'catA','cat2':'catA','cat3':'catA',
                'cat4':'catB',
                'cat5':'catC',
                'cat6':'catD','cat7':'catD',
                'cat9':'catE','cat10':'catE'}
    data_embed['big_category']=data_embed['category'].replace(cat_Bigcat)

    # ----------------- (4) SMALL & BIG category Indices----------------- #
    cat1 = [aspect_cols.index(i) for i in ['A민감성', 'A세정력', 'A촉촉함']]
    cat2 = [aspect_cols.index(i) for i in ['A민감성', 'A보습감', 'A향']]
    cat3 = [aspect_cols.index(i) for i in ['A민감성','A사용감', 'A발림성']]
    cat4 = [aspect_cols.index(i) for i in ['A커버력', 'A지속력', 'A피부톤']]
    cat5 = [aspect_cols.index(i) for i in ['A지속력', 'A피부톤', 'A촉촉함', 'A유분기', 'A발색감', 'A제형']]
    cat6 = [aspect_cols.index(i) for i in ['A지속력', 'A사용감', 'A발색감']]
    cat7 = [aspect_cols.index(i) for i in ['A지속력', 'A사용감', 'A발색감']]
    cat8 = [aspect_cols.index(i) for i in ['A향기', 'A민감성', 'A용량']]
    cat9 = [aspect_cols.index(i) for i in ['A향기', 'A민감성', 'A용량']]
    catA=list(set(cat1+cat2+cat3))
    catB=cat4  
    catC=cat5   
    catD=list(set(cat6+cat7))  
    catE=list(set(cat8+cat9))
    big_cat_indices=[catA,catB,catC,catD,catE]
    small_cat_indices=[cat1,cat2,cat3,cat4,cat5,cat6,cat7,cat8,cat9] 

    drop_idx = np.where( data_embed['embed_sents'].apply(len) == 0)[0].tolist()
    if len(drop_idx) != 0:
        data_embed.drop(drop_idx, inplace = True)
        #print('dropping empty rows:', drop_idx)

    drop_idx2 = list( data_embed[ data_embed['embed_sents'].apply(lambda x:np.array(sum(1*np.array([type(item)!= np.ndarray for item in x] ))))!= 0].index)
    data_embed.drop(drop_idx2, inplace = True)
    #print('Dropping nans...', drop_idx2)
    data_embed.reset_index(drop = True, inplace = True)
    return data_embed, aspect_cols, big_cat_indices, small_cat_indices

def three_masks(aspect_cols,catBig,catSmall):
    mask_mine=torch.zeros(len(aspect_cols))
    mask_notmine=torch.zeros(len(aspect_cols))
    mask_etc=torch.ones(len(aspect_cols))
    mask_mine[catSmall]=1
    mask_notmine[catBig]=1
    mask_notmine[catSmall]=0
    mask_etc[catBig]=0
    return mask_mine,mask_notmine,mask_etc

## temporary LOSS 위해 추가 (수정 가능)
def labeling(y1,y1_mask):
    lab = torch.zeros((len(y1_mask),3))
    lab[torch.where(y1==1)[0],2] = 1
    lab[torch.where(y1==-1)[0],0] = 1
    lab[torch.where(y1==0)[0],1] = 1
    lab[torch.where(y1_mask==0)[0],1] = 0
    return lab

## temporary LOSS 위해 추가 (수정 가능)
def label_rating(rating_value, n_cls):
    y2 = torch.zeros((n_cls))
    if rating_value == 5:
        y2[2] = 1
    elif rating_value == 4:
        y2[1] = 1
    elif rating_value == 1:
        y2[0] = 1
    return y2

####################  2. DATASET MODULE    ####################

class CustomDataset(Dataset):
    def __init__(self,config):
        self.config = config
        #data, aspect_cols, big_cat_indices, self.small_cat_indices = preprocess(config['data_path'], model_type=config['embed'])
        with open('amore_data_detected_bert.pkl', 'rb') as f:
            data = pickle.load(f)

        with open('aspect_cols.pkl', 'rb') as f:
            aspect_cols = pickle.load(f)

        with open('big_cat_indices.pkl', 'rb') as f:
            big_cat_indices = pickle.load(f)

        with open('small_cat_indices.pkl', 'rb') as f:
            self.small_cat_indices = pickle.load(f)

        self.catA,self.catB,self.catC,self.catD,self.catE = big_cat_indices
        self.x_sent=data['embed_sents']

        #self.x_sent=X_sent.values
        self.y1_data=data[list(data.columns[data.columns.str.contains('A')])].values
        self.y2_data=data['rating'].values
        self.y3_mask = data['mask'].values
        self.task1=data.index[data.category=='cat1']
        self.task2=data.index[data.category=='cat2']
        self.task3=data.index[data.category=='cat3']
        self.task4=data.index[data.category=='cat4']
        self.task5=data.index[data.category=='cat5']
        self.task6=data.index[data.category=='cat6']
        self.task7=data.index[data.category=='cat7']
        self.task8=data.index[data.category=='cat9']
        self.task9=data.index[data.category=='cat10']
        self.mask1=three_masks(aspect_cols,self.catA,self.small_cat_indices[0])
        self.mask2=three_masks(aspect_cols,self.catA,self.small_cat_indices[1])
        self.mask3=three_masks(aspect_cols,self.catA,self.small_cat_indices[2])
        self.mask4=three_masks(aspect_cols,self.catB,self.small_cat_indices[3])
        self.mask5=three_masks(aspect_cols,self.catC,self.small_cat_indices[4])
        self.mask6=three_masks(aspect_cols,self.catD,self.small_cat_indices[5])
        self.mask7=three_masks(aspect_cols,self.catD,self.small_cat_indices[6])
        self.mask8=three_masks(aspect_cols,self.catE,self.small_cat_indices[7])
        self.mask9=three_masks(aspect_cols,self.catE,self.small_cat_indices[8])


    def __len__(self):
        return len(self.x_sent)

    def __getitem__(self, index):
        #print(index)
        x_sent = torch.FloatTensor(self.x_sent[index])   
        y1 = torch.FloatTensor(self.y1_data[index])
        rating_value = torch.LongTensor([self.y2_data[index]])
        y2 = label_rating(rating_value, self.config['n_class_t'])
        y3 = self.y3_mask[index]
        
        mask_mine = torch.zeros(16)

        if index in self.task1:
            big_task=torch.tensor(1)
            small_task=torch.tensor(1)
            label = labeling(y1,self.mask1[0])
            mask_mine[ np.array(self.small_cat_indices[ 0 ])[y3] ] = 1
            
        elif index in self.task2:
            big_task=torch.tensor(1)
            small_task=torch.tensor(2)
            label = labeling(y1,self.mask2[0])
            mask_mine[ np.array(self.small_cat_indices[ 1 ])[y3] ] = 1
            
        elif index in self.task3:
            big_task=torch.tensor(1)
            small_task=torch.tensor(3)
            label = labeling(y1,self.mask3[0])
            mask_mine[ np.array(self.small_cat_indices[ 2 ])[y3] ] = 1
            
        elif index in self.task4:
            big_task=torch.tensor(2)
            small_task=torch.tensor(4)
            label = labeling(y1,self.mask4[0])
            mask_mine[ np.array(self.small_cat_indices[ 3 ])[y3] ] = 1

        elif index in self.task5:
            big_task=torch.tensor(3)
            small_task=torch.tensor(5)
            label = labeling(y1,self.mask5[0])
            mask_mine[ np.array(self.small_cat_indices[ 4 ])[y3] ] = 1
            
        elif index in self.task6:
            big_task=torch.tensor(4)
            small_task=torch.tensor(6)
            label = labeling(y1,self.mask6[0])
            mask_mine[ np.array(self.small_cat_indices[ 5 ])[y3] ] = 1
            
        elif index in self.task7:
            big_task=torch.tensor(4)
            small_task=torch.tensor(7)
            label = labeling(y1,self.mask7[0])
            mask_mine[ np.array(self.small_cat_indices[ 6 ])[y3] ] = 1
            
        elif index in self.task8:
            big_task=torch.tensor(5)
            small_task=torch.tensor(8)
            label = labeling(y1,self.mask8[0])
            mask_mine[ np.array(self.small_cat_indices[ 7 ])[y3] ] = 1
            
        elif index in self.task9:
            big_task=torch.tensor(5)
            small_task=torch.tensor(9)
            label = labeling(y1,self.mask9[0])
            mask_mine[ np.array(self.small_cat_indices[ 8 ])[y3] ] = 1

        return x_sent, y1, y2, mask_mine, big_task, small_task, label

def collate_fn(batch):
    x_sent, y1, y2, mask_mine, big_task, small_task, label = zip(*batch)
    x_sent = pad_sequence(x_sent, batch_first=True)
    mask_sent=x_sent.sum(axis=2)
    mask_sent[mask_sent!=0]=1
    x_sent_num=mask_sent.sum(axis=1).int()
    y1=torch.stack(y1)
    y2=torch.stack(y2)
    big_task = torch.stack(big_task)
    small_task = torch.stack(small_task)
    label = torch.stack(label)
    mask_mine=torch.stack(mask_mine)

    return x_sent, x_sent_num, y1, y2,mask_mine,big_task,small_task, label

####################  3. DATALOADER MODULE    ####################

from torch.utils.data import TensorDataset, random_split, RandomSampler, SequentialSampler, ConcatDataset, SubsetRandomSampler

def check_label_cnts(dataset, df):
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    small_cat_indices = dataset.small_cat_indices
    cat_lst = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat9', 'cat10']
    map_dict = dict( zip( cat_lst, small_cat_indices) )
    
    df_lst = []
    df_aspects = pd.concat([df.iloc[:,:16].astype(int), df.iloc[:,16] ], axis = 1 )

    for cat in cat_lst:
        df_tmp = df_aspects[df_aspects['category']== cat].iloc[:,map_dict[cat]].apply(pd.value_counts)
        df_tmp.fillna(0, inplace = True)
        df_tmp = df_tmp.astype(int)
        cols = [i[1] for i in df_tmp.columns.str.split('A')]
        df_tmp.columns = cols    
        df_lst.append( df_tmp.to_dict() )
    cat_dict = dict(zip(cat_lst, df_lst))
    cat_names = ['클렌징', '모이스쳐', '선케어', '페이스', '립', '아이' ,'기타색조(치크)', '바디', '헤어']
    
    # 향수 제거
    d_new = dict(zip(cat_lst, cat_names ))
    cat_new = dict( ( d_new[key], value) for (key, value) in cat_dict.items() )

    dict_tot = {}
    dict_tot['[1]스킨케어'] = dict(zip(list( cat_new.keys() )[:3], list( cat_new.values() )[:3]))
    dict_tot['[2]메이크업1'] = dict(zip(list( cat_new.keys() )[3:4], list( cat_new.values() )[3:4]))
    dict_tot['[3]메이크업2'] = dict(zip(list( cat_new.keys() )[4:5], list( cat_new.values() )[4:5]))
    dict_tot['[4]메이크업3'] = dict(zip(list( cat_new.keys() )[5:7], list( cat_new.values() )[5:7]))
    dict_tot['[5]생활용품'] = dict(zip(list( cat_new.keys() )[7:], list( cat_new.values() )[7:]))
    pp.pprint(dict_tot)


def aspect_based_split_ids(config, dataset,df):
    def split_indices(split_df, idx, config):
        #print(indices)
        indices = split_df[ split_df == idx].index
        tr_size = int( len(indices) * config['tr_ratio'] )
        val_size = int( len(indices) * config['val_ratio']  )
        test_size = len(indices) - tr_size - val_size
        print(f'Split ASP[{split_df.name}],LAB[{idx}] tr:{tr_size} val:{val_size} test: {test_size}')
        tr_idx = np.random.choice( indices, tr_size, replace = False)
        val_idx = np.random.choice( list( set(indices) - set(tr_idx) ), val_size, replace = False)
        test_idx = list( set(indices) - set(tr_idx) - set(val_idx) )
        return tr_idx, val_idx, test_idx
    tr_, val_, test_ = [], [], []

    df_aspects = pd.concat([df.iloc[:,:16].astype(int), df.iloc[:,16] ], axis = 1 )

    cat_lst = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat9', 'cat10']
    d_new = {'cat1': '클렌징','cat2': '모이스쳐','cat3': '선케어',
             'cat4': '페이스','cat5': '립','cat6': '아이','cat7': '기타색조(치크)',
             'cat9': '바디','cat10': '헤어'}

    small_cat_indices = dataset.small_cat_indices
    map_dict = dict( zip( cat_lst, small_cat_indices) )

    for idx, cat in enumerate(cat_lst):
        #print( d_new[cat] )
        df_tmp = df_aspects[df_aspects['category']== cat].iloc[:,map_dict[cat]]
        df_cnt = df_tmp.apply(pd.value_counts)
        print(idx, cat)
        if idx != 6:
            col_idx = np.argmax( df_cnt.iloc[np.where( df_cnt.index == -1)[0][0]] )
        else:
            col_idx = np.argmax( df_cnt.iloc[np.where( df_cnt.index == 0)[0][0]] )
        split_df = df_tmp.iloc[:,col_idx]

        for idx in set(split_df):
            tr_idx, val_idx, test_idx = split_indices( split_df, idx, config )
            tr_.extend(tr_idx) #tr_indices.extend(tr_idx)
            val_.extend(val_idx) #val_indices.extend(val_idx)
            test_.extend(test_idx) #test_indices.extend(test_idx)

    return tr_, val_, test_

## Train Valid Test Split (small task 의 ASPECT LABEL 가운데 -1 이 가장 많은 aspect의 -1, 0, 1 의 비율에 맞게)
def get_split_sampler(config, dataset, df):
    tr_ratio =config['tr_ratio'] # 0.6
    val_ratio = config['val_ratio'] # 0.2

    #print('='*80)
    #print('SPLITTING DATASET...')
    tr_indices, val_indices, test_indices = aspect_based_split_ids(config, dataset, df)

    train_sampler = SubsetRandomSampler(tr_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, val_sampler, test_sampler

def train_val_test_loader(config):    

    dataset = CustomDataset(config)
    with open('amore_data_detected_bert.pkl', 'rb') as f:
        df = pickle.load(f)
    #df, _, _, _ = preprocess(config['data_path'], model_type=config['embed'])

    config['check_label_cnts'] = True
    if config['check_label_cnts'] == True:
        print('='*80)
        print('\nCHECK LABEL COUNTS ====>')
        check_label_cnts(dataset, df)

    bs = config['batch_size']
    train_sampler, val_sampler, test_sampler = get_split_sampler(config, dataset, df)
    print('batch_size :', bs)

    tr_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=bs, sampler=train_sampler, collate_fn=collate_fn, num_workers = 2,  drop_last = True)
    val_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=bs, sampler=val_sampler, collate_fn=collate_fn, num_workers = 2, drop_last = True)
    test_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=bs, sampler=test_sampler, collate_fn=collate_fn)

    return tr_loader, val_loader, test_loader
