import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.utils import *
from src.utils.paths import get_model_dir, get_statistics_file, get_model_file
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, DataLoader
import numpy as np
import copy
import argparse
import os
from pathlib import Path


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 32
        self.lin0 = torch.nn.Linear(1, dim)

        nn = Sequential(Linear(2, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


def computeRankLoss(prediction, edge_attr, use_l2 = True):
    grads = torch.autograd.grad(outputs=prediction, inputs=edge_attr,
                                      grad_outputs=torch.ones(prediction.size()).cuda(), create_graph=True)
    grads = grads[0]
    lower_grads = F.relu(grads[:, 0])
    upper_grads = F.relu(-1*grads[:, 1])
    if use_l2:
        rank_loss = lower_grads.square().sum() + upper_grads.square().sum()
    else:
        rank_loss = lower_grads.sum() + upper_grads.sum()
    return rank_loss


def getArg():
    parser = argparse.ArgumentParser(description='Message Passing for MOCU Prediction')
    parser.add_argument('--pretrain', default='.',
                        help='pretrained model name')
    parser.add_argument('--name', default='MP_Graph',
                        help='file name to save the model')
    parser.add_argument('--data_path', default='',
                        help='')
    parser.add_argument('--EPOCH', default=400, type=int,
                        help='EPOCH to train')
    parser.add_argument('--test_only', action='store_true',
                        help='output test result only')
    parser.add_argument('--debug', action='store_true',
                        help='print debug information')
    parser.add_argument('--Constrain_weight', default=0.0001, type=float,
                        help='rank loss weight')
    parser.add_argument('--multiple_model', action='store_true',
                        help='use multiple models for test')
    parser.add_argument('--model_dir', default=None,
                        help='Directory to save/load models (default: ../models/name or ../experiments/name/models/)')

    args = parser.parse_args()
    return args


def loadData(test_only, data_path, pretrain, name, model_dir):

    print('Preparing data...')
    data_list = torch.load(data_path)

    if test_only:
        model_paths = pretrain.split('+')
        pretrain = model_paths[0]
        # Try model_dir first (from centralized paths), then fall back to old location
        stats_path = get_statistics_file(pretrain)
        if not stats_path.exists():
            # Fallback to old location for backward compatibility
            from pathlib import Path
            old_path = Path('../models') / pretrain / 'statistics.pth'
            if old_path.exists():
                stats_path = old_path
            else:
                stats_path = get_statistics_file(name)  # Try current experiment
        
        statistics = torch.load(str(stats_path))
        mean = statistics['mean']
        std = statistics['std']
        for d in data_list:
            d.y = (d.y - mean) / std
        data_test = data_list[0:len(data_list)]
        train_loader = []
        test_loader = DataLoader(data_test, batch_size=128, shuffle=False)

    else:
        mean = np.asarray([d.y[0][0] for d in data_list]).mean()
        std = np.asarray([d.y[0][0] for d in data_list]).std()
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = data_list[0:int(0.96 * len(data_list))]
        data_test = data_list[int(0.96 * len(data_list)):len(data_list)]
        train_loader = DataLoader(data_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(data_test, batch_size=128, shuffle=False)
        
        # Save statistics using centralized path utility
        stats_path = get_statistics_file(name)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'mean': mean, 'std': std}, str(stats_path))

    return train_loader, test_loader, [std, mean]


def main():

    print(sys.argv)
    args = getArg()
    EPOCH = args.EPOCH if not args.test_only else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine model directory - use centralized path utility
    if args.model_dir:
        # If explicitly provided (from run.sh), use it
        model_dir = Path(args.model_dir)
    else:
        # Otherwise use centralized path utility
        model_dir = get_model_dir(args.name)
    
    # Ensure directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, test_loader, [std, mean] = loadData(args.test_only, args.data_path, args.pretrain, args.name, str(model_dir))
    print('Making Model...')
    with torch.backends.cudnn.flags(enabled=False):
        model = Net().cuda()  # Create model FIRST
    
        if args.pretrain != '.':
            if args.multiple_model:
                model_paths = args.pretrain.split('+')
                models = []
                for model_path in model_paths:
                    m = Net().cuda()
                    # Use centralized path utility
                    pretrain_path = get_model_file(model_path)
                    if not pretrain_path.exists():
                        # Fallback to old location
                        pretrain_path = Path('../models') / model_path / 'model.pth'
                    m.load_state_dict(torch.load(str(pretrain_path)))
                    models.append(m)
            else:
                # Load pretrained weights into the model using centralized path utility
                pretrain_path = get_model_file(args.pretrain)
                if not pretrain_path.exists():
                    # Fallback to old location
                    pretrain_path = Path('../models') / args.pretrain / 'model.pth'
                model.load_state_dict(torch.load(str(pretrain_path)))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.7, patience=5, min_lr=0.00001)

        test_MSE = np.zeros(EPOCH)
        train_MSE = np.zeros(EPOCH)
        train_rank = np.zeros(EPOCH)
        # start training
        for epoch in range(EPOCH):
            if not args.test_only:
                train_MSE_step = []
                train_rank_step = []
                model.train()
                for data in train_loader:  # for each training step
                    # train
                    data_ = copy.deepcopy(data)
                    data_.edge_attr.requires_grad = True
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    data_ = data_.to(device)
                    optimizer.zero_grad()
                    prediction = model(data_).unsqueeze(dim=1)  # [batch_size, 1]
                    mseLoss = F.mse_loss(prediction, data_.y)
                    rankLoss = args.Constrain_weight*computeRankLoss(prediction, data_.edge_attr)
                    loss = mseLoss + rankLoss
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    train_MSE_step.append(mseLoss.item() * std * std)
                    train_rank_step.append(rankLoss.item() * std * std)

                train_MSE[epoch] = (sum(train_MSE_step) / len(train_MSE_step))
                train_rank[epoch] = (sum(train_rank_step) / len(train_rank_step))
                # Print progress every 10 epochs or at the end
                if (epoch + 1) % 10 == 0 or epoch == EPOCH - 1:
                    print('Epoch %d/%d | LR: %.6f | Train MSE: %.6f | Train Rank: %.6f' % 
                          (epoch + 1, EPOCH, lr, train_MSE[epoch], train_rank[epoch]))

            # test
            model.eval()
            error = 0
            for data in test_loader:
                data = data.to(device)
                if args.multiple_model:
                    predictions = 0
                    for model in models:
                        model.eval()
                        predictions = predictions + model(data).unsqueeze(dim=1)
                    prediction = predictions/len(models)
                else:
                    prediction = model(data).unsqueeze(dim=1)
                error += (prediction * std - data.y * std).square().sum().item()  # MSE
            loss = error / len(test_loader.dataset)
            test_MSE[epoch] = loss
            # Print test result every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == EPOCH - 1:
                print('         | Test MSE: %.6f' % loss)
            
            # Adaptive minimum epoch threshold: allow saving earlier for short training runs
            # For short runs (<=20 epochs): save from epoch 2+
            # For longer runs: save from epoch 5+ (allows initial warmup period)
            min_save_epoch = 2 if EPOCH <= 20 else 5
            
            # Save model if it's better than previous best, or if it's the last epoch (ensure model is always saved)
            should_save = False
            if epoch == EPOCH - 1:
                # Always save on the last epoch to ensure model exists
                should_save = True
                save_reason = "final epoch"
            elif epoch >= min_save_epoch and loss < min(test_MSE[0:epoch]):
                # Save if it's better than previous best (after minimum epoch threshold)
                should_save = True
                save_reason = "best so far"
            
            if should_save:
                # Use centralized path utility
                model_path = get_model_file(args.name)
                model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                torch.save(model.state_dict(), str(model_path))
                if save_reason == "best so far" and (epoch + 1) % 10 != 0:
                    print('         | Test MSE: %.6f (best, saved)' % loss)
                elif save_reason == "final epoch":
                    print('         | Model saved (final epoch)')

    # plot and save
    plotCurves(train_MSE, train_rank, test_MSE, EPOCH, args.name)

    # save some prediction result
    savePrediction(data, prediction, std, mean, args.name)

    if args.debug:
        printInstance(data, prediction, std, mean)


if __name__ == '__main__':
    main()
