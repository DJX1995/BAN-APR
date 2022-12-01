import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def forward(self, query_tokens, query_length):
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

        q_vector_list = []
        batch_size = query_length_.size(0)
        for i, length in enumerate(query_length_):
            hidden = output[i][0:length]
            q_vector = torch.mean(hidden, dim=0)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)
        return q_vector, output
        
        
class VisualEncoder(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=512, num_layers=1, bidirection=True):
        super(VisualEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.biLSTM = nn.LSTM(input_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)

    def forward(self, visual_data, visual_length, max_seq_len):
        visual_embedding = pack_padded_sequence(visual_data,
                                               visual_length.to('cpu').data.numpy(),
                                               batch_first=True,
                                               enforce_sorted=False)
        # h_0, c_0 is init as zero here
        output, _ = self.biLSTM(visual_embedding) # return (out, (h_n,c_n))
        # c_n and h_n: (num_directions, batch, hidden_size)
        # out: (batch, seq_len, num_directions, hidden_size)
        output, visual_length_ = pad_packed_sequence(output, batch_first=True, total_length=max_seq_len)

        v_vector_list = []
        batch_size = visual_length_.size(0)
        for i, length in enumerate(visual_length_):
            hidden = output[i][0:length]
            v_vector = torch.mean(hidden, dim=0)
            v_vector_list.append(v_vector)
        v_vector = torch.stack(v_vector_list)
        return v_vector, output