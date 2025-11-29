import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        # x: [batch_size, seq_len, input_size]
        out, hidden = self.rnn(x, hidden)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])       # 取序列最后一个时间步输出
        return out, hidden
    
    def init_hidden(self, batch_size):
        # 初始化隐藏状态，RNN 要求形状 [num_layers, batch_size, hidden_size]
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)