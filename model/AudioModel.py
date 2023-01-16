import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self , input_size , hidden_size ,num_layers , num_classes , device) -> None:
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc = nn.Linear(self.hidden_size * 2 , self.num_classes)
        self.blstm = nn.LSTM(input_size = self.input_size , hidden_size = self.hidden_size ,num_layers = self.num_layers , batch_first = True , bidirectional = True)
    
    
    def attn_layer(self , lstm_output):
        final_hidden_state = lstm_output[:,-1,:].unsqueeze(-1)
        attn_weights = torch.bmm(lstm_output , final_hidden_state)
        soft_attn_weights =  F.softmax(attn_weights , dim=1)
        output =  torch.bmm(lstm_output.transpose(1,2) , soft_attn_weights).squeeze(2)
        return output
    

    def forward(self , input):
        h0 = torch.empty((2 * self.num_layers, input.size(dim = 0), self.hidden_size)).to(self.device)
        nn.init.orthogonal_(h0)
        c0 = torch.empty((2 * self.num_layers, input.size(dim = 0), self.hidden_size)).to(self.device)
        nn.init.orthogonal_(c0)
        
        output, _ = self.blstm(input ,(h0 , c0))
        output_a = self.attn_layer(output)
        
        return output , output_a

    

    
        
        

    


