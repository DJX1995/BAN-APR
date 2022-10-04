import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# glove_emb = np.load(open('../data/subset_data/activitynet_captions_glove_embeds.npy','rb')) 

class QueryEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, embed_dim=300, num_layers=1, 
                 bidirection=True, pre_train_weights=None):
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pre_train_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_train_weights))
            self.embedding.weight.requires_grad = False
        self.biLSTM = nn.LSTM(embed_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)
#         self.textualAttention = TextualAttention()

    def forward(self, query_tokens, query_length):
        '''
        query_length: a list
        
        q_vector: sentence level feature, this one has filtered out the zero padding index
        output: hidden states of each step
        '''
        query_embedding = self.embedding(query_tokens)
        query_embedding = pack_padded_sequence(query_embedding,
                                               query_length.to('cpu').data.numpy(),
                                               batch_first=True,
                                               enforce_sorted=False)
        # h_0, c_0 is init as zero here
        
        output, _ = self.biLSTM(query_embedding) # return (out, (h_n,c_n))
        
        # c_n and h_n: (num_directions, batch, hidden_size)
        # out: (batch, seq_len, num_directions, hidden_size)
        output, query_length_ = pad_packed_sequence(output, batch_first=True)
        
        # select the hidden state of the last word individually, since the lengths of query are variable 
        # and we have pad zero to make the same len. We select the exact len output
        q_vector_list = []
        batch_size = query_length_.size(0)
        # q_vector is sentence feature, two ways: 1. 1st and last hidden concatenation; 2. pool over all hiddens
        # for i, length in enumerate(query_length_):
        #     h1 = output[i][0]
        #     hs = output[i][length - 1]
        #     q_vector = torch.cat((h1, hs), dim=-1)
        #     q_vector_list.append(q_vector)
        for i, length in enumerate(query_length_):
            hidden = output[i][0:length]
            q_vector = torch.mean(hidden, dim=0)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)
        # Note: the output here is zero-padded, we need slice the non-zero items for the following operations.
        return q_vector, output
        
        
class VisualEncoder(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=512, num_layers=1, bidirection=True):
        super(VisualEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.biLSTM = nn.LSTM(input_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)

    def forward(self, visual_data, visual_length, max_seq_len):
        '''
        query_length: a list
        
        v_vector: video clip level feature, this one has filtered out the zero padding index
        output: hidden states of each step
        '''
        # print(visual_data.to('cpu').device,visual_length.to('cpu').device)
        visual_embedding = pack_padded_sequence(visual_data,
                                               visual_length.to('cpu').data.numpy(),
                                               batch_first=True,
                                               enforce_sorted=False)
        # print(visual_embedding.device)
        # h_0, c_0 is init as zero here
        
        output, _ = self.biLSTM(visual_embedding) # return (out, (h_n,c_n))
        
        # c_n and h_n: (num_directions, batch, hidden_size)
        # out: (batch, seq_len, num_directions, hidden_size)
        output, visual_length_ = pad_packed_sequence(output, batch_first=True, total_length=max_seq_len)
        
        # select the hidden state of the last word individually, since the lengths of query are variable 
        # and we have pad zero to make the same len. We select the exact len output
        v_vector_list = []
        batch_size = visual_length_.size(0)
        # [head, tail] as video feature
        # for i, length in enumerate(visual_length_):
        #     h1 = output[i][0]
        #     hs = output[i][length - 1]
        #     v_vector = torch.cat((h1, hs), dim=-1)
        #     v_vector_list.append(v_vector)
        # v_vector = torch.stack(v_vector_list)
        # mean pool as video feature
        for i, length in enumerate(visual_length_):
            hidden = output[i][0:length]
            v_vector = torch.mean(hidden, dim=0)
            v_vector_list.append(v_vector)
        v_vector = torch.stack(v_vector_list)
        # Note: the output here is zero-padded, we need slice the non-zero items for the following operations.
        return v_vector, output