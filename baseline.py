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
from sklearn.metrics import precision_score, f1_score


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
parser.add_argument('--num_gradual', type = int, default = 5, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help = 'cicids', default = 'cicids')
parser.add_argument('--n_epoch', type=int, default=5)
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


class CICIDSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

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


def train(train_loader, model, optimizer, criterion, epoch):
    print('Training...')
    model.train()  # Set model to training mode
    train_total = 0
    train_correct = 0

    for i, (data, labels, _) in enumerate(train_loader):  # Corrected line
        data, labels = data.cuda(), labels.cuda()

        # Forward pass: Compute predicted outputs by passing inputs to the model
        logits = model(data)

        # Calculate the batch's accuracy
        _, predicted = torch.max(logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Calculate the loss
        loss = criterion(logits, labels)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1, len(train_loader), 100. * train_correct / train_total, loss.item()))

    train_acc = 100. * train_correct / train_total
    return train_acc

def evaluate(test_loader, model):
    print('Evaluating...')
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # For evaluation, we don't need gradients
        for data, labels, _ in test_loader:
            data, labels = data.cuda(), labels.cuda()
            logits = model(data)
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print(f'Test Accuracy: {accuracy:.4f} %')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, precision, f1



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

    # Create the PyTorch datasets with the SMOTE-applied variables
    train_dataset = CICIDSDataset(X_train_smote, y_train_smote)
    test_dataset = CICIDSDataset(X_test, y_test) # No SMOTE on test data

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

    # Define optimizers
    print(clf1.parameters)


    result_dir_path = os.path.dirname(txtfile)
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 test_acc1\n')

    epoch=0
    train_acc1=0
    # Evaluate models with initial (random) weights
    test_metrics = evaluate(test_loader, clf1)  # This now returns a tuple of three metrics
    print(f'Initial Evaluation - Accuracy: {test_metrics[0]:.4f}%, Precision: {test_metrics[1]:.4f}, F1 Score: {test_metrics[2]:.4f}')

    # Save initial evaluation results
    with open(txtfile, "a") as myfile:
        myfile.write(f"{epoch} {train_acc1:.4f} {test_metrics[0]:.4f} {test_metrics[1]:.4f} {test_metrics[2]:.4f}\n")

    # Training loop
    for epoch in range(1, args.n_epoch):
        # Train 
        clf1.train()
        adjust_learning_rate(optimizer1, epoch)
        criterion = nn.CrossEntropyLoss()  # Define the loss function

        train_acc1 = train(train_loader, clf1, optimizer1, criterion, epoch)
        
        # Evaluate model
        test_metrics = evaluate(test_loader, clf1)
        print(f'Epoch {epoch}/{args.n_epoch} - Test Accuracy: {test_metrics[0]:.4f}%, Precision: {test_metrics[1]:.4f}, F1 Score: {test_metrics[2]:.4f}')
        
        # Save evaluation results for current epoch
        with open(txtfile, "a") as myfile:
            myfile.write(f"{epoch} {train_acc1:.4f} {test_metrics[0]:.4f} {test_metrics[1]:.4f} {test_metrics[2]:.4f}\n")


if __name__=='__main__':
    main()
