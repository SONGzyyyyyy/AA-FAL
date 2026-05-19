import os
import sys
import torch
import yaml
import random
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models.mlp_head import MLPHead
from models.encoder_and_projection import Encoder_and_projection
from trainer import AAFALTrainer
from get_dataset import PreTrainDataset_prepared

RANDOM_SEED = 300

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(RANDOM_SEED)

    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config['trainer']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training with: {device}")

    X_train, Y_train = PreTrainDataset_prepared(
        params['ft'],
        range(params['class_start'], params['class_end'] + 1)
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=RANDOM_SEED
    )

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    freq_online_network = Encoder_and_projection(**config['network']).to(device)
    online_network = Encoder_and_projection(**config['network']).to(device)

    pred_params = {
        'in_channels': config['network']['projection_head']['projection_size'],
        **config['network']['projection_head']
    }
    freq_predictor = MLPHead(**pred_params).to(device)
    predictor = MLPHead(**pred_params).to(device)

    freq_target_network = Encoder_and_projection(**config['network']).to(device)
    target_network = Encoder_and_projection(**config['network']).to(device)

    freq_optimizer = torch.optim.Adam(
        list(freq_online_network.parameters()) + list(freq_predictor.parameters()),
        lr=params['lr']
    )
    optimizer = torch.optim.Adam(
        list(online_network.parameters()) + list(predictor.parameters()),
        lr=params['lr']
    )

    trainer = AAFALTrainer(
        freq_online_network=freq_online_network,
        freq_target_network=freq_target_network,
        freq_optimizer=freq_optimizer,
        freq_predictor=freq_predictor,
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        predictor=predictor,
        device=device,
        **config['trainer']
    )

    trainer.train_and_val(train_dataset, val_dataset)

if __name__ == '__main__':
    main()