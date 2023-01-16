import pickle
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import numpy as np

class audioDataset(Dataset):
    def __init__(self , path) -> None:
        super().__init__()
        f = open(path , 'rb')
        self.datas , self.labels , self.fileNames =  pickle.load(f)
        self.max_length = 750
        
        for i , mfcc in  enumerate(self.datas):
            if mfcc.shape[0] < self.max_length:
                padding = np.zeros((self.max_length - mfcc.shape[0] , mfcc.shape[1])) #(padding_length , 40)
                self.datas[i] = np.concatenate([mfcc , padding] , axis=0) #看concat
            elif mfcc.shape[0] > self.max_length:
                self.datas[i] = self.datas[i][:self.max_length]
            
        

    def __getitem__(self, index):
        le = preprocessing.LabelEncoder()
        target = le.fit_transform(self.labels)
        labels_Tensor = torch.from_numpy(target)
        datas_Tensor = torch.Tensor(self.datas[index])
        return  datas_Tensor , labels_Tensor[index]


    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    d = audioDataset("dataset/IEMOCAP_audio.pkl")
    print(d.__getitem__(0))
