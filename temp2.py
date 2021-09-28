import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

####################  1. LANGUAGE MODEL    ####################

# TO BE ADDED: BERT, HAN, Others


####################  2. MTL NETWORK W/ DOWNSTREAM TASKS    ####################

def create_layer(dropout, dim1, dim2):
    return nn.Sequential( nn.SELU(), nn.Dropout(dropout), 
            nn.Linear(dim1, dim2), nn.LayerNorm(dim2) )

def create_layer_small(dropout, dim1, dim2):
    return nn.Sequential( nn.SELU(), nn.Dropout(dropout), nn.Linear(dim1, dim2),
                          nn.LayerNorm(dim2), nn.Dropout(dropout), 
                          nn.Linear(dim2, 3), nn.SELU() )

class context_vector_3(nn.Module):
    def __init__(self, embed_dim):
        super(context_vector_3, self).__init__()
        self.cv1 = nn.Linear( embed_dim, 1, bias = False)
        self.cv2 = nn.Linear( embed_dim, 1, bias = False)
        self.cv3 = nn.Linear( embed_dim, 1, bias = False)
    def forward(self, x1, x2, x3):
        return self.cv1(x1), self.cv2(x2), self.cv3(x3)

class context_vector_6(nn.Module):
    def __init__(self, embed_dim):
        super(context_vector_6, self).__init__()
        self.cv1 = nn.Linear( embed_dim, 1, bias = False)
        self.cv2 = nn.Linear( embed_dim, 1, bias = False)
        self.cv3 = nn.Linear( embed_dim, 1, bias = False)
        self.cv4 = nn.Linear( embed_dim, 1, bias = False)
        self.cv5 = nn.Linear( embed_dim, 1, bias = False)
        self.cv6 = nn.Linear( embed_dim, 1, bias = False)

    def forward(self, x1, x2, x3, x4, x5, x6):
        return self.cv1(x1), self.cv2(x2), self.cv3(x3), self.cv4(x4), self.cv5(x5), self.cv6(x6)

class MultiSmallNet3(nn.Module):
    def __init__(self, config, task_dim):
        super(MultiSmallNet3, self).__init__()
        self.a1 = create_layer_small(config['drop_4'], task_dim, config['task_s'])
        self.a2 = create_layer_small(config['drop_4'], task_dim, config['task_s'])
        self.a3 = create_layer_small(config['drop_4'], task_dim, config['task_s'])

    def forward(self, x1, x2, x3):
        return torch.vstack( (self.a1(x1).unsqueeze(0), 
                              self.a2(x2).unsqueeze(0), 
                              self.a3(x3).unsqueeze(0) ) )

class MultiSmallNet6(nn.Module):
    def __init__(self, config, task_n):
        super(MultiSmallNet6, self).__init__()
        self.a1 = create_layer_small(config['drop_4'], task_n, config['task_s'])
        self.a2 = create_layer_small(config['drop_4'], task_n, config['task_s'])
        self.a3 = create_layer_small(config['drop_4'], task_n, config['task_s'])
        self.a4 = create_layer_small(config['drop_4'], task_n, config['task_s'])
        self.a5 = create_layer_small(config['drop_4'], task_n, config['task_s'])
        self.a6 = create_layer_small(config['drop_4'], task_n, config['task_s'])

    def forward(self, x1, x2, x3, x4, x5, x6):
        return torch.vstack( ( self.a1(x1).unsqueeze(0), 
                              self.a2(x2).unsqueeze(0), 
                              self.a3(x3).unsqueeze(0), 
                              self.a4(x4).unsqueeze(0), 
                              self.a5(x5).unsqueeze(0), 
                              self.a6(x6).unsqueeze(0) ) )

## Model Modules
class MultiABSA(nn.Module):
    ###########  1. CONSTRUCTOR  ###########

    def __init__(self, config):
        super(MultiABSA, self).__init__()
        # Shared Layer
        self.shared_layer1 = nn.Sequential(
            nn.Linear( 2 * config['embed_dim'] , config['shared_dim1'] ),  
            nn.Dropout( config['drop_1'] ), 
            nn.BatchNorm1d( config['shared_dim1'] ) )
        
        # Task 1: Overall
        self.task_overall =  nn.Sequential(
                    nn.SELU(),
                    nn.Linear( config['shared_dim1'], config['task_overall'] ), 
                    nn.BatchNorm1d( config['task_overall'] ), 
                    nn.Dropout( config['drop_2'] ), 
                    nn.Linear( config['task_overall'] , config['n_class_o']  ) 
                    ) 
        

        # Task 2: Task Specific

        # Sentence Module
        self.sent_gru = nn.GRU( config['embed_dim'], config['sent_gru_h_dim'], 
                                num_layers = config['sent_gru_n_layers'], batch_first = True, 
                                bidirectional = True, dropout = config['sent_gru_drop'] )
        self.sent_layer_norm = nn.LayerNorm( 2 * config['sent_gru_h_dim'], elementwise_affine = True)

        # attention matrix
        self.sent_attention1 = nn.Linear(2 * config['sent_gru_h_dim'], config['sent_att_dim'])
        self.sent_attention2 = nn.Linear(2 * config['sent_gru_h_dim'], config['sent_att_dim'])
        self.sent_attention3 = nn.Linear(2 * config['sent_gru_h_dim'], config['sent_att_dim'])
        self.sent_attention4 = nn.Linear(2 * config['sent_gru_h_dim'], config['sent_att_dim'])
        self.sent_attention5 = nn.Linear(2 * config['sent_gru_h_dim'], config['sent_att_dim'])

        self.group_attention1 = nn.Linear(2 * config['sent_gru_h_dim'], config['aspect_att_dim'])
        self.group_attention2 = nn.Linear(2 * config['sent_gru_h_dim'], config['aspect_att_dim'])
        self.group_attention3 = nn.Linear(2 * config['sent_gru_h_dim'], config['aspect_att_dim'])
        self.group_attention4 = nn.Linear(2 * config['sent_gru_h_dim'], config['aspect_att_dim'])
        self.group_attention5 = nn.Linear(2 * config['sent_gru_h_dim'], config['aspect_att_dim'])

        # task specific & product Context Vector for Attention
        self.task_cv1 = context_vector_3(config['sent_att_dim'])
        self.task_cv2 = context_vector_3(config['sent_att_dim'])
        self.task_cv3 = context_vector_3(config['sent_att_dim'])
        self.task_cv4 = context_vector_3(config['sent_att_dim'])
        self.task_cv5 = context_vector_6(config['sent_att_dim'])
        self.task_cv6 = context_vector_3(config['sent_att_dim'])
        self.task_cv7 = context_vector_3(config['sent_att_dim'])
        self.task_cv8 = context_vector_3(config['sent_att_dim'])
        self.task_cv9 = context_vector_3(config['sent_att_dim'])

        self.group_cv1 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv2 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv3 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv4 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv5 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv6 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv7 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv8 = nn.Linear( config['aspect_att_dim'], 1, bias = False)
        self.group_cv9 = nn.Linear( config['aspect_att_dim'], 1, bias = False)

        ## Task Specific Network - BIG (제품군)
        self.big_shared = create_layer(config['drop_2'], config['shared_dim1'], config['shared_dim2'])

        self.big_task1 = create_layer(config['drop_2'], config['shared_dim2'], config['shared_dim3'])
        self.big_task2 = create_layer(config['drop_2'], config['shared_dim2'], config['shared_dim3'])
        self.big_task3 = create_layer(config['drop_2'], config['shared_dim2'], config['shared_dim3'])
        self.big_task4 = create_layer(config['drop_2'], config['shared_dim2'], config['shared_dim3'])
        self.big_task5 = create_layer(config['drop_2'], config['shared_dim2'], config['shared_dim3'])

        ## TASK Specific Network - MID (제품군)
        self.mid_task1 = create_layer(config['drop_3'], config['shared_dim3'], config['task1'])
        self.mid_task2 = create_layer(config['drop_3'], config['shared_dim3'], config['task2'])
        self.mid_task3 = create_layer(config['drop_3'], config['shared_dim3'], config['task3'])
        self.mid_task4 = create_layer(config['drop_3'], config['shared_dim3'], config['task4'])
        self.mid_task5 = create_layer(config['drop_3'], config['shared_dim3'], config['task5'])
        self.mid_task6 = create_layer(config['drop_3'], config['shared_dim3'], config['task6'])
        self.mid_task7 = create_layer(config['drop_3'], config['shared_dim3'], config['task7'])
        self.mid_task8 = create_layer(config['drop_3'], config['shared_dim3'], config['task8'])
        self.mid_task9 = create_layer(config['drop_3'], config['shared_dim3'], config['task9'])

        ## Task Specific Network - SMALL (제품의 속성)
        self.small_ts1 = MultiSmallNet3(config, config['task1']) 
        self.small_ts2 = MultiSmallNet3(config, config['task2']) 
        self.small_ts3 = MultiSmallNet3(config, config['task3']) 
        self.small_ts4 = MultiSmallNet3(config, config['task4']) 
        self.small_ts5 = MultiSmallNet6(config, config['task5']) 
        self.small_ts6 = MultiSmallNet3(config, config['task6']) 
        self.small_ts7 = MultiSmallNet3(config, config['task7']) 
        self.small_ts8 = MultiSmallNet3(config, config['task8']) 
        self.small_ts9 = MultiSmallNet3(config, config['task9']) 
        
    ###########  2. MEMBER FUNCTION  ###########

    def calc_att (self, energy, valid_bs, *args):
        # SOFTMAX TYPE ATTENTION
        a_exp = torch.exp( energy - energy.max() )
        a_exp_pad, _ = pad_packed_sequence(PackedSequence(a_exp, valid_bs), batch_first = True)
        return a_exp_pad / torch.sum (a_exp_pad, dim = 1, keepdim = True ) 

    def get_att_vec3(self, task_cv, u, valid_bsz):
        att_u1 = self.calc_att(task_cv.cv1(u).squeeze(1), valid_bsz)
        att_u2 = self.calc_att(task_cv.cv2(u).squeeze(1), valid_bsz)
        att_u3 = self.calc_att(task_cv.cv3(u).squeeze(1), valid_bsz)
        return att_u1, att_u2, att_u3

    def get_att_vec6(self, task_cv, u, valid_bsz):
        att_u1 = self.calc_att(task_cv.cv1(u).squeeze(1), valid_bsz)
        att_u2 = self.calc_att(task_cv.cv2(u).squeeze(1), valid_bsz)
        att_u3 = self.calc_att(task_cv.cv3(u).squeeze(1), valid_bsz)
        att_u4 = self.calc_att(task_cv.cv4(u).squeeze(1), valid_bsz)
        att_u5 = self.calc_att(task_cv.cv5(u).squeeze(1), valid_bsz)
        att_u6 = self.calc_att(task_cv.cv6(u).squeeze(1), valid_bsz)
        return att_u1, att_u2, att_u3, att_u4, att_u5, att_u6

    def get_doc_vec3(self, h_i_pad, att1, att2, att3):
        doc_vec1 = (h_i_pad * att1.unsqueeze(2)).sum(dim = 1)
        doc_vec2 = (h_i_pad * att2.unsqueeze(2)).sum(dim = 1)
        doc_vec3 = (h_i_pad * att3.unsqueeze(2)).sum(dim = 1)
        return doc_vec1, doc_vec2, doc_vec3

    def get_doc_vec6(self, h_i_pad, att1, att2, att3, att4, att5, att6):
        doc_vec1 = (h_i_pad * att1.unsqueeze(2)).sum(dim = 1)
        doc_vec2 = (h_i_pad * att2.unsqueeze(2)).sum(dim = 1)
        doc_vec3 = (h_i_pad * att3.unsqueeze(2)).sum(dim = 1)
        doc_vec4 = (h_i_pad * att4.unsqueeze(2)).sum(dim = 1)
        doc_vec5 = (h_i_pad * att5.unsqueeze(2)).sum(dim = 1)
        doc_vec6 = (h_i_pad * att6.unsqueeze(2)).sum(dim = 1)
        return doc_vec1, doc_vec2, doc_vec3, doc_vec4, doc_vec5, doc_vec6

    def calc_prod_att3(self, doc1, doc2, doc3, group_att, global_cv):
        e1 = torch.tanh( global_cv( group_att(doc1) ) )
        e2 = torch.tanh( global_cv( group_att(doc2) ) )
        e3 = torch.tanh( global_cv( group_att(doc3) ) )
        energy = torch.hstack([e1,e2,e3])
        att_wt = torch.exp(energy) / torch.exp(energy).sum(axis = 1).unsqueeze(1)
        doc_sum = (torch.stack([doc1, doc2, doc3]) * att_wt.T.unsqueeze(2)).sum(axis = 0)
        return doc_sum, att_wt

    def calc_prod_att6(self, doc1, doc2, doc3, doc4, doc5, doc6, group_att, global_cv):
        e1 = torch.tanh( global_cv( group_att(doc1) ) )
        e2 = torch.tanh( global_cv( group_att(doc2) ) )
        e3 = torch.tanh( global_cv( group_att(doc3) ) )
        e4 = torch.tanh( global_cv( group_att(doc4) ) )
        e5 = torch.tanh( global_cv( group_att(doc5) ) )
        e6 = torch.tanh( global_cv( group_att(doc6) ) )
        energy = torch.hstack([e1,e2,e3,e4,e5,e6])
        att_wt = torch.exp(energy) / torch.exp(energy).sum(axis = 1).unsqueeze(1)
        doc_sum = (torch.stack([doc1, doc2, doc3, doc4, doc5, doc6]) * att_wt.T.unsqueeze(2)).sum(axis = 0)
        return doc_sum, att_wt

    ###########  3. FORWARD METHOD  ###########

    def forward (self, docs, doc_lengths):
        
        # ----------------- (1) Attention ----------------- #
        # packing
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim = 0, descending = True)
        docs = docs[doc_perm_idx]
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first = True)
        valid_bsz_sent = packed_sents.batch_sizes

        # autoregressive mapping
        h_i, _ = self.sent_gru(packed_sents)
        h_i_normed = self.sent_layer_norm( h_i.data )
        h_i_pad, _ = pad_packed_sequence( h_i, batch_first = True )

        # calculate attention & document vector
        u_1 = torch.tanh(self.sent_attention1(h_i_normed.data))
        u_2 = torch.tanh(self.sent_attention2(h_i_normed.data))
        u_3 = torch.tanh(self.sent_attention3(h_i_normed.data))
        u_4 = torch.tanh(self.sent_attention4(h_i_normed.data))
        u_5 = torch.tanh(self.sent_attention5(h_i_normed.data))

        # attention for doc_vec
        att11, att12, att13 = self.get_att_vec3(self.task_cv1, u_1, valid_bsz_sent )
        att21, att22, att23 = self.get_att_vec3(self.task_cv2, u_1, valid_bsz_sent )
        att31, att32, att33 = self.get_att_vec3(self.task_cv3, u_1, valid_bsz_sent )
        att41, att42, att43 = self.get_att_vec3(self.task_cv4, u_2, valid_bsz_sent )
        att51, att52, att53, att54, att55, att56 = self.get_att_vec6(self.task_cv5, 
                                                               u_3, valid_bsz_sent )
        att61, att62, att63 = self.get_att_vec3(self.task_cv6, u_4, valid_bsz_sent )
        att71, att72, att73 = self.get_att_vec3(self.task_cv7, u_4, valid_bsz_sent )
        att81, att82, att83 = self.get_att_vec3(self.task_cv8, u_5, valid_bsz_sent )
        att91, att92, att93 = self.get_att_vec3(self.task_cv9, u_5, valid_bsz_sent )
        
        # doc_vec (for each aspect)
        doc11, doc12, doc13 = self.get_doc_vec3(h_i_pad, att11, att12, att13)
        doc21, doc22, doc23 = self.get_doc_vec3(h_i_pad, att21, att22, att23)
        doc31, doc32, doc33 = self.get_doc_vec3(h_i_pad, att31, att32, att33)
        doc41, doc42, doc43 = self.get_doc_vec3(h_i_pad, att41, att42, att43)
        doc51, doc52, doc53, doc54, doc55, doc56 = self.get_doc_vec6(h_i_pad, 
                                                att51, att52, att53, att54, att55, att56)
        doc61, doc62, doc63 = self.get_doc_vec3(h_i_pad, att61, att62, att63)
        doc71, doc72, doc73 = self.get_doc_vec3(h_i_pad, att71, att72, att73)
        doc81, doc82, doc83 = self.get_doc_vec3(h_i_pad, att81, att82, att83)
        doc91, doc92, doc93 = self.get_doc_vec3(h_i_pad, att91, att92, att93)

        # group attention (for general aspect)
        doc1, att1 = self.calc_prod_att3(doc11, doc12, doc13, self.group_attention1, self.group_cv1)
        doc2, att2 = self.calc_prod_att3(doc21, doc22, doc23, self.group_attention1, self.group_cv2)
        doc3, att3 = self.calc_prod_att3(doc31, doc32, doc33, self.group_attention1, self.group_cv3)
        doc4, att4 = self.calc_prod_att3(doc41, doc42, doc43, self.group_attention2, self.group_cv4)
        doc5, att5 = self.calc_prod_att6(doc51, doc52, doc53, doc54, doc55, doc56,
                                                              self.group_attention3, self.group_cv5)
        doc6, att6 = self.calc_prod_att3(doc61, doc62, doc63, self.group_attention4, self.group_cv6)
        doc7, att7 = self.calc_prod_att3(doc71, doc72, doc73, self.group_attention4, self.group_cv7)
        doc8, att8 = self.calc_prod_att3(doc81, doc82, doc83, self.group_attention5, self.group_cv8)
        doc9, att9 = self.calc_prod_att3(doc91, doc92, doc93, self.group_attention5, self.group_cv9)

        # Reorder
        _, doc_unperm_idx = doc_perm_idx.sort(dim = 0, descending = False)
        att11, att12, att13 = att11[doc_unperm_idx], att12[doc_unperm_idx], att13[doc_unperm_idx]
        att21, att22, att23 = att21[doc_unperm_idx], att22[doc_unperm_idx], att23[doc_unperm_idx]
        att31, att32, att33 = att31[doc_unperm_idx], att32[doc_unperm_idx], att33[doc_unperm_idx]
        att41, att42, att43 = att41[doc_unperm_idx], att42[doc_unperm_idx], att43[doc_unperm_idx]
        att51, att52, att53 = att51[doc_unperm_idx], att52[doc_unperm_idx], att53[doc_unperm_idx]
        att54, att55, att56 = att54[doc_unperm_idx], att56[doc_unperm_idx], att56[doc_unperm_idx]
        att61, att62, att63 = att61[doc_unperm_idx], att62[doc_unperm_idx], att63[doc_unperm_idx]
        att71, att72, att73 = att71[doc_unperm_idx], att72[doc_unperm_idx], att73[doc_unperm_idx]
        att81, att82, att83 = att81[doc_unperm_idx], att82[doc_unperm_idx], att83[doc_unperm_idx]
        att91, att92, att93 = att91[doc_unperm_idx], att92[doc_unperm_idx], att93[doc_unperm_idx]

        att1, att2, att3 = att1[doc_unperm_idx], att2[doc_unperm_idx], att3[doc_unperm_idx]
        att4, att5, att6 = att4[doc_unperm_idx], att5[doc_unperm_idx], att6[doc_unperm_idx]
        att7, att8, att9 = att7[doc_unperm_idx], att8[doc_unperm_idx], att9[doc_unperm_idx]


        # ----------------- (2) MultiTaskNet ----------------- #
        # TASK A: general score
        tg1 = self.task_overall( self.shared_layer1( doc1[doc_unperm_idx] ) )
        tg2 = self.task_overall( self.shared_layer1( doc2[doc_unperm_idx] ) )
        tg3 = self.task_overall( self.shared_layer1( doc3[doc_unperm_idx] ) )
        tg4 = self.task_overall( self.shared_layer1( doc4[doc_unperm_idx] ) )
        tg5 = self.task_overall( self.shared_layer1( doc5[doc_unperm_idx] ) )
        tg6 = self.task_overall( self.shared_layer1( doc6[doc_unperm_idx] ) )
        tg7 = self.task_overall( self.shared_layer1( doc7[doc_unperm_idx] ) )
        tg8 = self.task_overall( self.shared_layer1( doc8[doc_unperm_idx] ) )
        tg9 = self.task_overall( self.shared_layer1( doc9[doc_unperm_idx] ) )
        
        # TASK B: task specific score
        ts11 = self.mid_task1( self.big_task1( self.big_shared( self.shared_layer1( doc11[doc_unperm_idx] ) ) ) )
        ts12 = self.mid_task1( self.big_task1( self.big_shared( self.shared_layer1( doc12[doc_unperm_idx] ) ) ) )
        ts13 = self.mid_task1( self.big_task1( self.big_shared( self.shared_layer1( doc13[doc_unperm_idx] ) ) ) )
        ts21 = self.mid_task2( self.big_task1( self.big_shared( self.shared_layer1( doc21[doc_unperm_idx] ) ) ) )
        ts22 = self.mid_task2( self.big_task1( self.big_shared( self.shared_layer1( doc22[doc_unperm_idx] ) ) ) )
        ts23 = self.mid_task2( self.big_task1( self.big_shared( self.shared_layer1( doc23[doc_unperm_idx] ) ) ) )
        ts31 = self.mid_task3( self.big_task1( self.big_shared( self.shared_layer1( doc31[doc_unperm_idx] ) ) ) )
        ts32 = self.mid_task3( self.big_task1( self.big_shared( self.shared_layer1( doc32[doc_unperm_idx] ) ) ) )
        ts33 = self.mid_task3( self.big_task1( self.big_shared( self.shared_layer1( doc33[doc_unperm_idx] ) ) ) )

        ts41 = self.mid_task4( self.big_task2( self.big_shared( self.shared_layer1( doc41[doc_unperm_idx] ) ) ) )
        ts42 = self.mid_task4( self.big_task2( self.big_shared( self.shared_layer1( doc42[doc_unperm_idx] ) ) ) )
        ts43 = self.mid_task4( self.big_task2( self.big_shared( self.shared_layer1( doc43[doc_unperm_idx] ) ) ) )
        
        ts51 = self.mid_task5( self.big_task3( self.big_shared( self.shared_layer1( doc51[doc_unperm_idx] ) ) ) )
        ts52 = self.mid_task5( self.big_task3( self.big_shared( self.shared_layer1( doc52[doc_unperm_idx] ) ) ) )
        ts53 = self.mid_task5( self.big_task3( self.big_shared( self.shared_layer1( doc53[doc_unperm_idx] ) ) ) )
        ts54 = self.mid_task5( self.big_task3( self.big_shared( self.shared_layer1( doc54[doc_unperm_idx] ) ) ) )
        ts55 = self.mid_task5( self.big_task3( self.big_shared( self.shared_layer1( doc55[doc_unperm_idx] ) ) ) )
        ts56 = self.mid_task5( self.big_task3( self.big_shared( self.shared_layer1( doc56[doc_unperm_idx] ) ) ) )
        
        ts61 = self.mid_task6( self.big_task4( self.big_shared( self.shared_layer1( doc61[doc_unperm_idx] ) ) ) )
        ts62 = self.mid_task6( self.big_task4( self.big_shared( self.shared_layer1( doc62[doc_unperm_idx] ) ) ) )
        ts63 = self.mid_task6( self.big_task4( self.big_shared( self.shared_layer1( doc63[doc_unperm_idx] ) ) ) )
        ts71 = self.mid_task7( self.big_task4( self.big_shared( self.shared_layer1( doc71[doc_unperm_idx] ) ) ) )
        ts72 = self.mid_task7( self.big_task4( self.big_shared( self.shared_layer1( doc72[doc_unperm_idx] ) ) ) )
        ts73 = self.mid_task7( self.big_task4( self.big_shared( self.shared_layer1( doc73[doc_unperm_idx] ) ) ) )
        
        ts81 = self.mid_task8( self.big_task5( self.big_shared( self.shared_layer1( doc81[doc_unperm_idx] ) ) ) )
        ts82 = self.mid_task8( self.big_task5( self.big_shared( self.shared_layer1( doc82[doc_unperm_idx] ) ) ) )
        ts83 = self.mid_task8( self.big_task5( self.big_shared( self.shared_layer1( doc83[doc_unperm_idx] ) ) ) )
        ts91 = self.mid_task9( self.big_task5( self.big_shared( self.shared_layer1( doc91[doc_unperm_idx] ) ) ) )
        ts92 = self.mid_task9( self.big_task5( self.big_shared( self.shared_layer1( doc92[doc_unperm_idx] ) ) ) )
        ts93 = self.mid_task9( self.big_task5( self.big_shared( self.shared_layer1( doc93[doc_unperm_idx] ) ) ) )

        ts1 = self.small_ts1( ts11, ts12, ts13 )
        ts2 = self.small_ts2( ts21, ts22, ts23 )
        ts3 = self.small_ts3( ts31, ts32, ts33 )
        ts4 = self.small_ts4( ts41, ts42, ts43 )
        ts5 = self.small_ts5( ts51, ts52, ts53, ts54, ts55, ts56 )
        ts6 = self.small_ts6( ts61, ts62, ts63 )
        ts7 = self.small_ts7( ts71, ts72, ts73 )
        ts8 = self.small_ts8( ts81, ts82, ts83 )
        ts9 = self.small_ts9( ts91, ts92, ts93 )

        return ((tg1, tg2, tg3, tg4, tg5, tg6, tg7, tg8, tg9), 
                (ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9), 
                (att11, att12, att13, att21, att22, att23, att31, att32, att33), 
                (att41, att42, att43), (att51, att52, att53, att54, att55, att56), 
                (att61, att62, att63, att71, att72, att73), 
                (att81, att82, att83, att91, att92, att93), 
                (att1, att2, att3, att4, att5, att6, att7, att8, att9) )

####### attention 관련 추가 모듈 

# def initialize_attention(self, config ):
#     W = nn.init.xavier_uniform_( nn.Parameter( torch.zeros( (config['doc_dim'],config['shared_dim2']) ) ) )
#     stdv = 1 / np.sqrt( W.size(1) )
#     b = nn.Parameter( torch.tensor( np.random.uniform(-stdv, stdv) ) )
#     return W, b

# self.task0_att_w, self.task0_att_b = self.initialize_attention(config)
# self.task1_att_w, self.task1_att_b = self.initialize_attention(config)
# self.task2_att_w, self.task2_att_b = self.initialize_attention(config)
# self.task3_att_w, self.task3_att_b = self.initialize_attention(config)
# self.task4_att_w, self.task4_att_b = self.initialize_attention(config)
# self.task5_att_w, self.task5_att_b = self.initialize_attention(config)