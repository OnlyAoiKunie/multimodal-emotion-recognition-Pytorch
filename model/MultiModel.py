import torch
import torch.nn as nn
from model.AudioModel import *
from model.TextModel import *
import torch.nn.functional as F
class multiModel(nn.Module):
    def __init__(self , input_size_audio , input_size_text , hidden_size ,num_layers , num_classes , device) -> None:
        super().__init__()
        self.audio_model = AudioEncoder(input_size_audio , hidden_size ,num_layers , num_classes , device)
        self.text_model = TextEncoder(input_size_text , hidden_size ,num_layers , num_classes , device)

        self.fc = nn.Linear(hidden_size * 4 , num_classes)
        

    def cross_attn(self , lstm_output_key , lstm_attn_query):  # key : (b , s , h)  query : (b , h) 
        q = lstm_attn_query.unsqueeze(-1)
        att = torch.bmm(lstm_output_key , q)
        att_weight = F.softmax(att , dim = 1) #[b , s , 1]
        output = torch.bmm(lstm_output_key.transpose(1,2) , att_weight).squeeze(2)
        return output
        

    def forward(self , text_input , text_input_mask , audio_input):
        output_text , output_text_attn = self.text_model(text_input , text_input_mask) 
        output_audio , output_audio_attn = self.audio_model(audio_input)
        audio_to_text = self.cross_attn(output_text , output_audio_attn)
        text_to_audio = self.cross_attn(output_audio , output_text_attn)
        output = torch.cat([audio_to_text , text_to_audio] , dim = 1) # [batch , 800] [batch , 400] [batch , 400]
        output = self.fc(output)
        
        return output

    

    
        
        

    


