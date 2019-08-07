import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import metrics.counting_metrics as cm
import os
import math
import copy

from counting_datasets.CityCam import CityCam
from counting_datasets.CityCam_maker import time_stamp
from counting_datasets.CityCam import ToTensor
from models.mcnn import MCNN
import hyper_param_conf as hp 

# Global variables
use_gpu = torch.cuda.is_available()
dataset_dir = 'CityCam/'
hyper_params = f'Epochs-{hp.epochs}_BatchSize-{hp.batch_size}_LR-{hp.learning_rate}_Momentum-{hp.momentum}_Version-{hp.version}'
checkpoint_dir = os.path.join('checkpoints', hyper_params)
device = torch.device('cuda:0' if use_gpu else 'cpu')
# Global variables

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

def train(pretrained=None):
    """
    Train the counting model.
    Args:
        pretrained: Indicates if you want to load a pretrained model or train a
                    new model from scratch. When loading pretrained parameters,
                    this argument should be the path of the checkpoint.
    """
    model = MCNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params=model.parameters(),
    lr=hp.learning_rate,
    momentum=hp.momentum)

    data_trans = transforms.Compose((ToTensor(),))
    dataset = {phase: CityCam(root_dir=dataset_dir,
                                dataset_type=phase,
                                transform=data_trans)
                for phase in ('Train', 'Test')}
    
    data_loader = {
        'train': DataLoader(dataset['Train'], batch_size=hp.batch_size, shuffle=True),
        'val': DataLoader(dataset['Test'], batch_size=hp.batch_size, shuffle=False)
    }

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

    if pretrained:
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

    if use_gpu:
        model.to(device)

    loss_list = []
    mae_list = []
    mse_list = []
    min_mae = 9999999
    best_model_weights = copy.deepcopy(model.state_dict())
    
    try:
        for epoch in range(hp.epochs):
            print(f'{time_stamp()} - Epoch {epoch + 1}/{hp.epochs}')
            print(f'{time_stamp()}' + '-' * 28)

            for phase in ('train', 'val'):
                if phase == 'train':
                    model.train()
                    print(f'{time_stamp()} - training...')
                else:
                    model.eval()
                    print(f'{time_stamp()} - validating...')

                epoch_loss = 0
                epoch_mae = 0
                epoch_mse = 0
                for idx, batch in enumerate(data_loader[phase]):
                    img = batch['image']
                    gt_dm = batch['density_map']
                    gt_count = batch['gt_count']

                    img = img.to(device)
                    gt_dm = gt_dm.to(device)

                    # zero-grad
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        et_dm = model(img)
                        et_dm = torch.squeeze(et_dm, dim=1)
                        down_sample = nn.Sequential(nn.MaxPool2d(2), nn.MaxPool2d(2))
                        down_gt_dm = down_sample(gt_dm)

                        loss = criterion(down_gt_dm, et_dm)
                        epoch_loss += loss.item()
                        epoch_mae += cm.mae(et_dm, gt_dm).item()
                        epoch_mse += cm.mse(et_dm, gt_dm).item()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        if (idx + 1) % 50 == 0:
                            print(f'{time_stamp()} - ' + 'Batch {}: running loss = {:.4f}, running AE = {:.4f}, running SE = {:.4f}'.format(
                                idx + 1, epoch_loss, epoch_mae, epoch_mse))
                
                mean_epoch_loss = epoch_loss / len(data_loader[phase])
                epoch_mae = epoch_mae / len(data_loader[phase])
                epoch_mse = math.sqrt(epoch_mse / len(data_loader[phase]))
                loss_list.append(mean_epoch_loss)
                mae_list.append(epoch_mae)
                mse_list.append(epoch_mse)
                
                print(f'{time_stamp()} - ' + 'Epoch {} - {}: epoch loss = {:.4f}, MAE = {:.4f}, MSE = {:.4f}'.format(
                    epoch + 1, phase, mean_epoch_loss, epoch_mae, epoch_mse
                ))

                if phase == 'val' and (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'loss': mean_epoch_loss,
                        'mae': epoch_mae
                    }, os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt'))
                    print(f'{time_stamp()} - Saved the model at epoch {epoch + 1}.')

                if phase == 'val' and epoch_mae < min_mae:
                    min_mae = epoch_mae
                    best_model_weights = copy.deepcopy(model.state_dict())
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'loss': mean_epoch_loss,
                        'mae': epoch_mae
                    }, os.path.join(checkpoint_dir, f'best_model.pt'))
                    print(f'{time_stamp()} - Saved the best model at epoch {epoch + 1}.')
    
    except KeyboardInterrupt:
        model.load_state_dict(best_model_weights)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': mean_epoch_loss,
            'mae': epoch_mae
        }, os.path.join(checkpoint_dir, f'best_model.pt'))
        print(f'Saved the best model at epoch {epoch + 1}.')
        quit()


if __name__ == "__main__":
    train()