# -*- coding:utf-8 -*-
from __future__ import print_function 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import MLPNet
import argparse, sys
import numpy as np
import datetime
from tqdm import tqdm
from loss import loss_coteaching, loss_coteaching_plus
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from torch.optim import AdamW


for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None 


import os


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.0001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help = 'cicids', default = 'cicids')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--optimizer', type = str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=500)
parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type = str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
parser.add_argument('--fr_type', type = str, help='forget rate type', default='type_1')
args = parser.parse_args()


# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 
init_epoch = 0



def introduce_label_noise(labels, noise_rate=args.noise_rate):
    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
    noisy_indices = np.random.choice(np.arange(n_samples), size=n_noisy, replace=False)

    # Initialize as all False, indicating no sample is noisy initially
    noise_or_not = np.zeros(n_samples, dtype=bool)  

    # Iterate over the randomly selected indices to introduce noise
    unique_labels = np.unique(labels)
    for idx in noisy_indices:
        original_label = labels[idx]
        # Exclude the original label to ensure the new label is indeed different
        possible_labels = np.delete(unique_labels, np.where(unique_labels == original_label))
        # Randomly select a new label from the remaining possible labels
        new_label = np.random.choice(possible_labels)
        labels[idx] = new_label  # Assign the new label
        noise_or_not[idx] = True  # Mark this index as noisy

    return labels, noise_or_not

class CICIDSDataset(Dataset):
    def __init__(self, features, labels, noise_or_not):
        self.features = features
        self.labels = labels
        self.noise_or_not = noise_or_not 

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return feature, label, index  

    def __len__(self):
        return len(self.labels)

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) 
       
# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type=='type_1':
        rate_schedule = np.ones(args.n_epoch)*forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    return rate_schedule

rate_schedule = gen_forget_rate(args.fr_type)
  
save_dir = args.result_dir +'/' +args.dataset+'/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate)

txtfile = save_dir + "/" + model_str + ".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_dataset, train_loader, epoch, model1, optimizer1, model2, optimizer2, noise_or_not):
    print('Training %s...' % model_str)
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0

    for i, (data, labels, indices) in enumerate(train_loader):

        ind=indices.cpu().numpy().transpose()
      
        labels = Variable(labels).cuda()
        
        if args.dataset=='news':
            data = Variable(data.long()).cuda()
        else:
            data = Variable(data).cuda()
        # Forward + Backward + Optimize
        logits1=model1(data)
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(data)
        prec2,  = accuracy(logits2, labels, topk=(1, ))
        train_total2+=1
        train_correct2+=prec2
        if epoch < init_epoch:
            loss_1, loss_2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        else:
            if args.model_type=='coteaching_plus':
                loss_1, loss_2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, epoch*i)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        if args.dataset=='news':
            data = Variable(data.long()).cuda()
        else:
            data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        if args.dataset=='news':
            data = Variable(data.long()).cuda()
        else:
            data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def weights_init(m):
    if isinstance(m, nn.Linear):
        # Apply custom initialization to linear layers
        init.xavier_uniform_(m.weight.data)  # Xavier initialization for linear layers
        if m.bias is not None:
            init.constant_(m.bias.data, 0)    # Initialize bias to zero

    elif isinstance(m, nn.Conv2d):
        # Apply custom initialization to convolutional layers
        init.kaiming_normal_(m.weight.data)   # Kaiming initialization for convolutional layers
        if m.bias is not None:
            init.constant_(m.bias.data, 0)     # Initialize bias to zero

def main():

    preprocessed_file_path = 'final_dataframe.csv'

    if os.path.exists(preprocessed_file_path):
        print("Concatonated dataset already exists")
        # Load the preprocessed DataFrame from the saved CSV file
        df = pd.read_csv(preprocessed_file_path)
    else:
        print("Dataset doesn't exists: generating...")

        df1 = pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        df2=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        df3=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")
        df4=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
        df5=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        df6=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        df7=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")
        df8=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

        # nRowsRead = 20000  # specify 'None' to read all rows
        # df6 = pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", nrows=nRowsRead)
        # df = pd.concat([df6], ignore_index=True)

        df.reset_index(drop=True, inplace=True)
        nRow, nCol = df.shape
        df.columns = df.columns.str.strip()
        print("Saving concatonated data")

        df.to_csv(preprocessed_file_path, index=False)

    if os.path.exists('numpy_data/features.npy'):
        print("Preprocessed data exists, loading...")
        features_np_standardized = np.load('numpy_data/features.npy')
        labels_np = np.load('numpy_data/labels.npy')
        indices = np.load('numpy_data/indices.npy')
        X_train_smote = np.load('numpy_data/X_train.npy')
        y_train_smote = np.load('numpy_data/y_train.npy')
        X_test = np.load('numpy_data/X_test.npy')
        y_test = np.load('numpy_data/y_test.npy')
        test_indices = np.load('numpy_data/test_indices.npy')  
    else:
        print("Preprocessed data doesn't exists, generating...")

        # Convert your DataFrame columns to numpy arrays if not already done
        features_np = df.drop('Label', axis=1).values.astype(np.float32)  
        labels_np = LabelEncoder().fit_transform(df['Label'].values)

        # Check for inf and -inf values
        print("Contains inf: ", np.isinf(features_np).any())
        print("Contains -inf: ", np.isneginf(features_np).any())

        # Check for NaN values
        print("Contains NaN: ", np.isnan(features_np).any())

        # Replace inf/-inf with NaN
        features_np[np.isinf(features_np) | np.isneginf(features_np)] = np.nan

        # Impute NaN values with the median of each column
        imputer = SimpleImputer(strategy='median')
        features_np_imputed = imputer.fit_transform(features_np)

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit on the imputed features data and transform it to standardize
        features_np_standardized = scaler.fit_transform(features_np_imputed)

        # Generate indices for your dataset, which will be used for splitting
        indices = np.arange(len(labels_np))

        # Split indices into training and testing sets
        train_indices, test_indices, y_train, y_test = train_test_split(indices, labels_np, test_size=0.3, random_state=42)

        # Correctly split the standardized and imputed dataset
        X_train, X_test, y_train, y_test = train_test_split(features_np_standardized, labels_np, test_size=0.3, random_state=42)

        # Apply SMOTE on the correctly preprocessed training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


        # Introduce label and feature noise AFTER splitting
        labels_noisy, noise_or_not = introduce_label_noise(labels_np, noise_rate=0.2) 


        # Save the numpy arrays
        print("Saving preprocessd data")

        np.save('numpy_data/features.npy', features_np_standardized)
        np.save('numpy_data/labels.npy', labels_np)
        np.save('numpy_data/indices.npy', indices)
        np.save('numpy_data/X_train.npy', X_train_smote)
        np.save('numpy_data/y_train.npy', y_train_smote)
        np.save('numpy_data/X_test.npy', X_test)
        np.save('numpy_data/y_test.npy', y_test)
        np.save('numpy_data/test_indices.npy', test_indices)


    noise_or_not = np.zeros(len(y_train_smote), dtype=bool)  

    # Create the PyTorch datasets with the SMOTE-applied variables
    train_dataset = CICIDSDataset(X_train_smote, y_train_smote, noise_or_not)
    test_dataset = CICIDSDataset(X_test, y_test, noise_or_not[test_indices]) # No SMOTE on test data

    noise_or_not = train_dataset.noise_or_not

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')

    torch.manual_seed(234565678)
    clf1 = MLPNet()
    clf1.apply(weights_init)
    clf1.cuda()
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)


    torch.manual_seed(45456678)  # Change the seed to get a different initialization
    clf2 = MLPNet()
    clf2.apply(weights_init)

    # Move models to GPU
    clf2.cuda()

    # Define optimizers
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)
    print(clf1.parameters)

    print(clf2.parameters)

    result_dir_path = os.path.dirname(txtfile)
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 train_acc2 test_acc1 test_acc2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2)  + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train 
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train(train_dataset, train_loader, epoch, clf1, optimizer1, clf2, optimizer2, noise_or_not)
        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + "\n")

if __name__=='__main__':
    main()
