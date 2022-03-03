import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

####################  1. LANGUAGE MODEL    ####################

# TO BE ADDED: BERT, HAN, Others


####################  2. MTL NETWORK W/ DOWNSTREAM TASKS    ####################

def create_layer(dropout, dim1, dim2):
    return nn.Sequential( nn.Linear(dim1, dim2),
                          nn.LayerNorm(dim2),
                          nn.SELU(), 
                          nn.Dropout(dropout),
                          )

def create_layer_2(dropout, dim1, dim2, dim3, dim4): #, dim5):
    return nn.Sequential( nn.Linear(dim1, dim2),
                          nn.LayerNorm(dim2),
                          nn.SELU(), 
                          nn.Dropout(dropout),
                          nn.Linear(dim2, dim3),
                          nn.LayerNorm(dim3),
                          nn.SELU(), 
                          nn.Dropout(dropout),
                          nn.Linear(dim3, dim4),
                          nn.LayerNorm(dim4),
                          nn.SELU(), 
                          nn.Dropout(dropout),
                          nn.Linear(dim4, 3)
                        )

def create_layer_small(dropout, dim1, dim2):
    return nn.Sequential( nn.Linear(dim1, dim2),
                          nn.LayerNorm(dim2), 
                          nn.SELU(), 
                          nn.Dropout(dropout),
                          nn.Linear(dim2, 3)
                          )

class context_vector_3(nn.Module):
    def __init__(self, embed_dim):
        super(context_vector_3, self).__init__()
        self.cv = nn.ModuleDict( { f'cv{i+1}': nn.Linear( embed_dim, 1, bias = False) for i in range(3)} )   
        
    def forward(self, x1, x2, x3):
        return self.cv.cv1(x1), self.cv.cv2(x2), self.cv.cv3(x3) #self.cv1(x1), self.cv2(x2), self.cv3(x3)

class context_vector_6(nn.Module):
    def __init__(self, embed_dim):
        super(context_vector_6, self).__init__()
        self.cv = nn.ModuleDict( { f'cv{i+1}': nn.Linear( embed_dim, 1, bias = False) for i in range(6)} )
        
    def forward(self, x1, x2, x3, x4, x5, x6):
        return self.cv.v1(x1), self.cv.cv2(x2), self.cv.cv3(x3), self.cv.cv4(x4), self.cv.cv5(x5), self.cv.cv6(x6)

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
        
        self.layer1 = nn.ModuleDict({f'shared{i+1}': nn.Sequential(
            nn.Linear( 2 * config['embed_dim'] , config['shared_dim1'] ),  
            nn.BatchNorm1d( config['shared_dim1'] ), 
            nn.SELU(),
            nn.Dropout( config['drop_1'] ), 
             )  for i in range(5)  })

        # Task 1: Overall
        self.task_overall = nn.ModuleDict({f'shared{i+1}': create_layer_2( config['drop_1'], 
                                            config['shared_dim1'], config['shared_dim2'],
                                            config['shared_dim3'], config['shared_dim4']) for i in range(5) })
        
        # Task 2: Task Specific  
        # Sentence Module
        self.sent_gru = nn.ModuleDict({ f'cat{i}': nn.GRU( config['embed_dim'], config['sent_gru_h_dim'], 
                                num_layers = config['sent_gru_n_layers'], batch_first = True, 
                                bidirectional = True, dropout = config['sent_gru_drop'] ) for i in range (1,6) })
        self.sent_layer_norm = nn.ModuleDict({ f'cat{i}':nn.LayerNorm( 2 * config['sent_gru_h_dim'], 
                                elementwise_affine = True) for i in range (1,6) })

        # attention matrix
        self.sent_att = nn.ModuleDict({ f'cat{i}':nn.Linear(2 * config['sent_gru_h_dim'], config['sent_att_dim'])
                                       for i in range (1, 6) })
        self.group_att = nn.ModuleDict({ f'cat{i}':nn.Linear(2 * config['sent_gru_h_dim'], config['aspect_att_dim'])
                                       for i in range (1, 6) })
        
        # task specific & product Context Vector for Attention

        self.task_cv = nn.ModuleDict({ f'task{i}':context_vector_3(config['sent_att_dim']) if i != 5 else 
                                        context_vector_6(config['sent_att_dim'])
                                       for i in range (1, 10) })
        self.group_cv = nn.ModuleDict({ f'task{i}':nn.Linear( config['aspect_att_dim'], 1, bias = False)
                                       for i in range (1, 10) })

        ## Task Specific Network - BIG (제품군)
        self.big_cat =  nn.ModuleDict({ f'cat{i}':create_layer(config['drop_2'], config['shared_dim1'], config['shared_dim2'])
                                       for i in range (1, 6) })

        ## TASK Specific Network - MID (제품군)
        self.mid_cat = nn.ModuleDict({ f'task{i}':create_layer(config['drop_3'], config['shared_dim2'], config[f'task{i}'])
                                       for i in range (1, 10) })
        
        ## Task Specific Network - SMALL (제품의 속성)
        self.aspect = nn.ModuleDict({ f'task{i}':MultiSmallNet3(config, config[f'task{i}']) if i!=5 else 
                        MultiSmallNet6(config, config['task5'])  
                                       for i in range (1, 10) })
        
    ###########  2. MEMBER FUNCTION  ###########

    def calc_att (self, energy, valid_bs, *args):
        # SOFTMAX TYPE ATTENTION
        a_exp = torch.exp( energy - energy.max() )
        a_exp_pad, _ = pad_packed_sequence(PackedSequence(a_exp, valid_bs), batch_first = True)
        return a_exp_pad / torch.sum (a_exp_pad, dim = 1, keepdim = True ) 

    def get_att_vec3(self, task_cv, u, valid_bsz):
        #task_cv.cv.
        att_u1 = self.calc_att(task_cv.cv.cv1(u).squeeze(1), valid_bsz)
        att_u2 = self.calc_att(task_cv.cv.cv2(u).squeeze(1), valid_bsz)
        att_u3 = self.calc_att(task_cv.cv.cv3(u).squeeze(1), valid_bsz)
        return att_u1, att_u2, att_u3

    def get_att_vec6(self, task_cv, u, valid_bsz):
        att_u1 = self.calc_att(task_cv.cv.cv1(u).squeeze(1), valid_bsz)
        att_u2 = self.calc_att(task_cv.cv.cv2(u).squeeze(1), valid_bsz)
        att_u3 = self.calc_att(task_cv.cv.cv3(u).squeeze(1), valid_bsz)
        att_u4 = self.calc_att(task_cv.cv.cv4(u).squeeze(1), valid_bsz)
        att_u5 = self.calc_att(task_cv.cv.cv5(u).squeeze(1), valid_bsz)
        att_u6 = self.calc_att(task_cv.cv.cv6(u).squeeze(1), valid_bsz)
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

    ###########  3. FORWARD PROPAGATION  ###########

    def forward (self, docs, doc_lengths):
        
        # ----------------- (1) Attention ----------------- #
        # packing
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim = 0, descending = True)
        docs = docs[doc_perm_idx]
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first = True)
        valid_bsz_sent = packed_sents.batch_sizes

        # autoregressive mapping
        u_is = []
        for i in range(1, 6):
            h_i, _ = self.sent_gru[f'cat{i}'](packed_sents)
            h_i_normed = self.sent_layer_norm[f'cat{i}'](h_i.data)
            u_is.append( torch.tanh(self.sent_att[f'cat{i}'](h_i_normed.data)))
            
        h_i_pad, _ = pad_packed_sequence( h_i, batch_first = True )
        
        # attention for doc_vec
        att11, att12, att13 = self.get_att_vec3(self.task_cv.task1, u_is[0], valid_bsz_sent )
        att21, att22, att23 = self.get_att_vec3(self.task_cv.task2, u_is[0], valid_bsz_sent )
        att31, att32, att33 = self.get_att_vec3(self.task_cv.task3, u_is[0], valid_bsz_sent )

        att41, att42, att43 = self.get_att_vec3(self.task_cv.task4, u_is[1], valid_bsz_sent )

        att51, att52, att53, att54, att55, att56 = self.get_att_vec6(self.task_cv.task5, 
                                                               u_is[2], valid_bsz_sent )

        att61, att62, att63 = self.get_att_vec3(self.task_cv.task6, u_is[3], valid_bsz_sent )
        att71, att72, att73 = self.get_att_vec3(self.task_cv.task7, u_is[3], valid_bsz_sent )

        att81, att82, att83 = self.get_att_vec3(self.task_cv.task8, u_is[4], valid_bsz_sent )
        att91, att92, att93 = self.get_att_vec3(self.task_cv.task9, u_is[4], valid_bsz_sent )
        
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
        doc1, att1 = self.calc_prod_att3(doc11, doc12, doc13, self.group_att.cat1, self.group_cv.task1)
        doc2, att2 = self.calc_prod_att3(doc21, doc22, doc23, self.group_att.cat1, self.group_cv.task2)
        doc3, att3 = self.calc_prod_att3(doc31, doc32, doc33, self.group_att.cat1, self.group_cv.task3)
        doc4, att4 = self.calc_prod_att3(doc41, doc42, doc43, self.group_att.cat2, self.group_cv.task4)
        doc5, att5 = self.calc_prod_att6(doc51, doc52, doc53, doc54, doc55, doc56,
                                                              self.group_att.cat3, self.group_cv.task5)
        doc6, att6 = self.calc_prod_att3(doc61, doc62, doc63, self.group_att.cat4, self.group_cv.task6)
        doc7, att7 = self.calc_prod_att3(doc71, doc72, doc73, self.group_att.cat4, self.group_cv.task7)
        doc8, att8 = self.calc_prod_att3(doc81, doc82, doc83, self.group_att.cat5, self.group_cv.task8)
        doc9, att9 = self.calc_prod_att3(doc91, doc92, doc93, self.group_att.cat5, self.group_cv.task9)

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
        tg1 = self.task_overall.shared1( self.layer1.shared1( doc1[doc_unperm_idx] ) )
        tg2 = self.task_overall.shared1( self.layer1.shared1( doc2[doc_unperm_idx] ) )
        tg3 = self.task_overall.shared1( self.layer1.shared1( doc3[doc_unperm_idx] ) )

        tg4 = self.task_overall.shared2( self.layer1.shared2( doc4[doc_unperm_idx] ) )

        tg5 = self.task_overall.shared3( self.layer1.shared3( doc5[doc_unperm_idx] ) )

        tg6 = self.task_overall.shared4( self.layer1.shared4( doc6[doc_unperm_idx] ) )
        tg7 = self.task_overall.shared4( self.layer1.shared4( doc7[doc_unperm_idx] ) )

        tg8 = self.task_overall.shared5( self.layer1.shared5( doc8[doc_unperm_idx] ) )
        tg9 = self.task_overall.shared5( self.layer1.shared5( doc9[doc_unperm_idx] ) )


        # TASK B: task specific score
        ts11 = self.mid_cat.task1( self.big_cat.cat1(self.layer1.shared1(doc11[doc_unperm_idx])))
        ts12 = self.mid_cat.task1( self.big_cat.cat1(self.layer1.shared1(doc12[doc_unperm_idx])))
        ts13 = self.mid_cat.task1( self.big_cat.cat1(self.layer1.shared1(doc13[doc_unperm_idx])))
        ts21 = self.mid_cat.task2( self.big_cat.cat1(self.layer1.shared1(doc21[doc_unperm_idx])))
        ts22 = self.mid_cat.task2( self.big_cat.cat1(self.layer1.shared1(doc22[doc_unperm_idx])))
        ts23 = self.mid_cat.task2( self.big_cat.cat1(self.layer1.shared1(doc23[doc_unperm_idx])))
        ts31 = self.mid_cat.task3( self.big_cat.cat1(self.layer1.shared1(doc31[doc_unperm_idx])))
        ts32 = self.mid_cat.task3( self.big_cat.cat1(self.layer1.shared1(doc32[doc_unperm_idx])))
        ts33 = self.mid_cat.task3( self.big_cat.cat1(self.layer1.shared1(doc33[doc_unperm_idx])))

        ts41 = self.mid_cat.task4( self.big_cat.cat2(self.layer1.shared2(doc41[doc_unperm_idx])))
        ts42 = self.mid_cat.task4( self.big_cat.cat2(self.layer1.shared2(doc42[doc_unperm_idx])))
        ts43 = self.mid_cat.task4( self.big_cat.cat2(self.layer1.shared2(doc43[doc_unperm_idx])))
        
        ts51 = self.mid_cat.task5( self.big_cat.cat3(self.layer1.shared3(doc51[doc_unperm_idx])))
        ts52 = self.mid_cat.task5( self.big_cat.cat3(self.layer1.shared3(doc52[doc_unperm_idx])))
        ts53 = self.mid_cat.task5( self.big_cat.cat3(self.layer1.shared3(doc53[doc_unperm_idx])))
        ts54 = self.mid_cat.task5( self.big_cat.cat3(self.layer1.shared3(doc54[doc_unperm_idx])))
        ts55 = self.mid_cat.task5( self.big_cat.cat3(self.layer1.shared3(doc55[doc_unperm_idx])))
        ts56 = self.mid_cat.task5( self.big_cat.cat3(self.layer1.shared3(doc56[doc_unperm_idx])))
        
        ts61 = self.mid_cat.task6( self.big_cat.cat4(self.layer1.shared4(doc61[doc_unperm_idx])))
        ts62 = self.mid_cat.task6( self.big_cat.cat4(self.layer1.shared4(doc62[doc_unperm_idx])))
        ts63 = self.mid_cat.task6( self.big_cat.cat4(self.layer1.shared4(doc63[doc_unperm_idx])))
        ts71 = self.mid_cat.task7( self.big_cat.cat4(self.layer1.shared4(doc71[doc_unperm_idx])))
        ts72 = self.mid_cat.task7( self.big_cat.cat4(self.layer1.shared4(doc72[doc_unperm_idx])))
        ts73 = self.mid_cat.task7( self.big_cat.cat4(self.layer1.shared4(doc73[doc_unperm_idx])))
        
        ts81 = self.mid_cat.task8( self.big_cat.cat5(self.layer1.shared5(doc81[doc_unperm_idx]))) 
        ts82 = self.mid_cat.task8( self.big_cat.cat5(self.layer1.shared5(doc82[doc_unperm_idx])))
        ts83 = self.mid_cat.task8( self.big_cat.cat5(self.layer1.shared5(doc83[doc_unperm_idx])))
        ts91 = self.mid_cat.task9( self.big_cat.cat5(self.layer1.shared5(doc91[doc_unperm_idx])))
        ts92 = self.mid_cat.task9( self.big_cat.cat5(self.layer1.shared5(doc92[doc_unperm_idx])))
        ts93 = self.mid_cat.task9( self.big_cat.cat5(self.layer1.shared5(doc93[doc_unperm_idx])))

        ts1 = self.aspect.task1( ts11, ts12, ts13 )
        ts2 = self.aspect.task2( ts21, ts22, ts23 )
        ts3 = self.aspect.task3( ts31, ts32, ts33 )
        ts4 = self.aspect.task4( ts41, ts42, ts43 )
        ts5 = self.aspect.task5( ts51, ts52, ts53, ts54, ts55, ts56 )
        ts6 = self.aspect.task6( ts61, ts62, ts63 )
        ts7 = self.aspect.task7( ts71, ts72, ts73 )
        ts8 = self.aspect.task8( ts81, ts82, ts83 )
        ts9 = self.aspect.task9( ts91, ts92, ts93 )

        return ((tg1, tg2, tg3, tg4, tg5, tg6, tg7, tg8, tg9), 
                (ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9), 
                (att11, att12, att13, att21, att22, att23, att31, att32, att33), 
                (att41, att42, att43), (att51, att52, att53, att54, att55, att56), 
                (att61, att62, att63, att71, att72, att73), 
                (att81, att82, att83, att91, att92, att93), 
                (att1, att2, att3, att4, att5, att6, att7, att8, att9) )