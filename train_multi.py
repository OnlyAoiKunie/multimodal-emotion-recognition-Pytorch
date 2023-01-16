from dataset.dataset_audio import *
from dataset.dataset_text import *
from dataset.dataset_multi import *
from model.TextModel import *
from model.AudioModel import *
from model.MultiModel import *
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
torch.cuda.set_device(2)
writer = SummaryWriter('./summary')
device =  torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def train(train_set, dev_set ,model ,batch_size , epoch_nums , lr):

    dataLoader_multi = DataLoader(train_set , batch_size=batch_size , shuffle=True)
    dataLoader_multi_dev = DataLoader(dev_set , batch_size=batch_size , shuffle=False)
    loss = nn.CrossEntropyLoss()
    minLoss_val = 0
    optimizer = torch.optim.Adam(model.parameters() , lr = lr)
    if torch.cuda.is_available():
        model.to(device)
        loss = loss.to(device)
    
    for epoch_num in range(epoch_nums):
        total_acc = 0
        total_loss = 0
        print("Epoch {} train start".format(epoch_num + 1))
        for train_data_audio , train_data_text in tqdm(dataLoader_multi):
            train_data_audio[0] = train_data_audio[0].to(device)
            train_data_audio[1] = train_data_audio[1].to(device)

            train_data_text[0] = train_data_text[0].to(device)
            train_data_text[1] = train_data_text[1].to(device)
            

            output = model(train_data_text[0] , train_data_text[1] , train_data_audio[0])
            batch_loss = loss(output , train_data_audio[1])
            total_loss += batch_loss.item()
            batch_acc = (output.argmax(dim=1) == train_data_audio[1]).sum().item()
            total_acc += batch_acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        print("Epoch {} train end | Train Acc: {} | Train Loss: {}".format(epoch_num + 1 , total_acc / train_set.__len__() , total_loss / train_set.__len__()))
        
        with torch.no_grad():
            total_acc_dev = 0
            total_loss_dev = 0
            print("Epoch {} validation start".format(epoch_num + 1))
            for dev_data_audio , dev_data_text in tqdm(dataLoader_multi_dev):

                dev_data_audio[0] = dev_data_audio[0].to(device)
                dev_data_audio[1] = dev_data_audio[1].to(device)

                dev_data_text[0] = dev_data_text[0].to(device)
                dev_data_text[1] = dev_data_text[1].to(device)
                

                output = model(dev_data_text[0] , dev_data_text[1], dev_data_audio[0])
                batch_loss_dev = loss(output , dev_data_audio[1])
                total_loss_dev += batch_loss_dev.item()
                batch_acc_dev = (output.argmax(dim=1) == dev_data_audio[1]).sum().item()
                total_acc_dev += batch_acc_dev
            if total_acc_dev / dev_set.__len__() > minLoss_val :
                minLoss_val = total_acc_dev / dev_set.__len__()
                torch.save(model.state_dict() , 'best_dev_model.pt')
                print("Validation best loss: {}，Save model".format(minLoss_val))
            writer.add_scalars('acc' , {'train':total_acc / train_set.__len__(),'dev':total_acc_dev / dev_set.__len__()} ,global_step=epoch_num + 1)
            writer.add_scalars('loss' , {'train':total_loss / train_set.__len__(),'dev':total_loss_dev / dev_set.__len__()} ,global_step=epoch_num + 1)
            print("Epoch {} validation end | Val Acc: {} | Val Loss: {}".format(epoch_num + 1 , total_acc_dev / dev_set.__len__() , total_loss_dev / dev_set.__len__()))



def test(test_set,model ,batch_size):
    dataLoader_audio_test = DataLoader(test_set , batch_size=batch_size , shuffle=False)
    model.load_state_dict(torch.load("best_dev_model.pt"))
    with torch.no_grad():
            total_acc_test = 0
            print("Test start")
            for test_data_audio , test_data_text in tqdm(dataLoader_audio_test):
                test_data_audio[0] = test_data_audio[0].to(device)
                test_data_audio[1] = test_data_audio[1].to(device)

                test_data_text[0] = test_data_text[0].to(device)
                test_data_text[1] = test_data_text[1].to(device)
                
                output = model(test_data_text[0] , test_data_text[1] , test_data_audio[0])
                
                batch_acc_test = (output.argmax(dim=1) == test_data_audio[1]).sum().item()
                total_acc_test += batch_acc_test
            print("Test end | Test Acc: {}".format(total_acc_test / test_set.__len__()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lr' , type=float , default=1e-3)
    p.add_argument('--batch_size' , type=int , default=32)
    p.add_argument('--epoch' , type=int , default=50)
    p.add_argument('--hidden_size' , type=int , default=200)
    p.add_argument('--input_size_text' , type=int , default=768)
    p.add_argument('--input_size_audio' , type=int , default=39)
    p.add_argument('--num_layers'  , type=int , default=1)
    p.add_argument('--device' , type=int , default=1)

    args = p.parse_args()

    dataset_text = textDataset("dataset/IEMOCAP_text.pkl")
    dataset_audio = audioDataset("dataset/IEMOCAP_audio.pkl")
    dataset_multi = multiDataset(dataset_audio , dataset_text)

    train_size = int(0.8 * dataset_multi.__len__())
    test_size = int(0.15 * dataset_multi.__len__())
    dev_size = dataset_multi.__len__() - train_size - test_size

    train_set_multi, test_set_multi, dev_set_multi = random_split(dataset_multi, [train_size, test_size, dev_size])
    

    print("Train size:{} | Validation size:{} | Test size:{}".format(train_set_multi.__len__() , dev_set_multi.__len__() , test_set_multi.__len__()))

    
    Model = multiModel(input_size_audio=args.input_size_audio, 
                        input_size_text=args.input_size_text, 
                        hidden_size=args.hidden_size, 
                        num_layers=args.num_layers, 
                        num_classes=4, 
                        device=device)

    train(train_set_multi, dev_set_multi , Model, args.batch_size, args.epoch , args.lr)
    test(test_set_multi , Model , args.batch_size)