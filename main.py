import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from counting_datasets.CityCam import CityCam
from counting_datasets.CityCam import ToTensor
from models.mcnn import MCNN
import hyper_param_conf as hp 

# Global variables
use_gpu = torch.cuda.is_available()
dataset_dir = 'CityCam/'
hyper_params = f'Epochs-{hp.epochs}_LR-{hp.learning_rate}_Momentum-{hp.momentum}_Version-{hp.version}'
checkpoint_dir = os.path.join('checkpoints', hyper_params)
device = torch.device('cuda:0' if use_gpu else 'cpu')
# Global variables

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

if not os.path.exists(checkpoint_dir):
    os.mkdir(os.path.join('checkpoints', checkpoint_dir))

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
        'train': DataLoader(dataset['Train'], batch_size=6, shuffle=True)
        'test': DataLoader(dataset['Test'], batch_size=6, shuffle=False)
    }

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

    if pretrained:
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

    if use_gpu:
        model.to(device)

    for epoch in range(hp.epochs):
        print(f'Epoch {epoch + 1}/{hp.epochs}')
        print('-' * 28)

        for phase in ('train', 'test'):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in data_loader[phase]:
                pass