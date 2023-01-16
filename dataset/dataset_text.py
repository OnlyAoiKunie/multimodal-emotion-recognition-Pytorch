import pickle
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from transformers import BertTokenizer
class textDataset(Dataset):
    def __init__(self , path) -> None:
        super().__init__()
        f = open(path , 'rb')
        self.datas , self.labels , self.fileNames =  pickle.load(f)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokens = tokenizer(self.datas , padding='max_length' , max_length = 128 ,  truncation=True , return_tensors = "pt")

    def __getitem__(self, index):
        le = preprocessing.LabelEncoder()
        target = le.fit_transform(self.labels)
        labels_Tensor = torch.from_numpy(target)
        return self.tokens['input_ids'][index] , self.tokens['attention_mask'][index] , labels_Tensor[index]


    def __len__(self):
        return len(self.datas)

if __name__ == '__main__':
    d = textDataset("dataset/IEMOCAP_text.pkl")
    print(d.__getitem__(5))