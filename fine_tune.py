import os
import sys
import torch
import yaml
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from get_dataset import FineTuneDataset_prepared

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_similarity_loss(features, target, device):
    features_norm = F.normalize(features, p=2, dim=1)
    cosine_sim = torch.mm(features_norm, features_norm.t())
    target = target.view(-1, 1)
    label_mask = torch.eq(target, target.T).float().to(device)
    pos_loss = - cosine_sim * label_mask
    neg_loss = cosine_sim * (1 - label_mask)
    return pos_loss.mean() + neg_loss.mean()

def train(online_network, classifier, loss_nll, train_dataloader, optim_online_network,
          optimizer_classifier, epoch, device, writer):
    online_network.train()
    classifier.train()
    correct = 0
    total_loss = 0.0
    for data, target in train_dataloader:
        data, target = data.to(device), target.long().to(device)
        optim_online_network.zero_grad()
        optimizer_classifier.zero_grad()
        features = online_network(data)[0]
        output = F.log_softmax(classifier(features), dim=1)
        nll = loss_nll(output, target)
        cos = cosine_similarity_loss(features, target, device)
        loss = nll + cos
        loss.backward()
        optim_online_network.step()
        optimizer_classifier.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(train_dataloader)
    acc = 100.0 * correct / len(train_dataloader.dataset)
    print(f'Train Epoch: {epoch} \tLoss: {total_loss:.6f}, Accuracy: {correct}/{len(train_dataloader.dataset)} ({acc:.2f}%)')
    writer.add_scalar('Accuracy/train', acc, epoch)
    writer.add_scalar('Loss/train', total_loss, epoch)
    return acc, total_loss

def evaluate(online_network, classifier, loss_nll, val_dataloader, epoch, device, writer):
    online_network.eval()
    classifier.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.long().to(device)
            output = F.log_softmax(classifier(online_network(data)[0]), dim=1)
            total_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(val_dataloader)
    acc = 100.0 * correct / len(val_dataloader.dataset)
    print(f'Validation set: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(val_dataloader.dataset)} ({acc:.2f}%)')
    writer.add_scalar('Accuracy/val', acc, epoch)
    writer.add_scalar('Loss/val', total_loss, epoch)
    return acc, total_loss

def test(online_network, classifier, loss_nll, test_dataloader, device):
    online_network.eval()
    classifier.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.long().to(device)
            output = F.log_softmax(classifier(online_network(data)[0]), dim=1)
            total_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(test_dataloader)
    acc = 100.0 * correct / len(test_dataloader.dataset)
    print(f'Test set: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({acc:.2f}%)')
    return acc

def train_and_test(online_network, classifier, loss_nll, train_dataloader, val_dataloader, test_dataloader,
                   optim_online_network, optim_classifier, scheduler_online_network, scheduler_classifier,
                   epochs, save_path_online_network, save_path_classifier, device, writer):
    current_min_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train(online_network, classifier, loss_nll, train_dataloader,
                                      optim_online_network, optim_classifier, epoch, device, writer)
        val_acc, val_loss = evaluate(online_network, classifier, loss_nll, val_dataloader, epoch, device, writer)
        if val_loss < current_min_test_loss:
            print(f"The validation loss improved from {current_min_test_loss:.6f} to {val_loss:.6f}, saving model.")
            current_min_test_loss = val_loss
            torch.save(online_network, save_path_online_network)
            torch.save(classifier, save_path_classifier)
        else:
            print("The validation loss did not improve.")
        print("------------------------------------------------")
    torch.save(online_network, 'Base_fintune/online_network.pth')
    torch.save(classifier, 'Base_fintune/classifier.pth')
    online_network = torch.load(save_path_online_network)
    classifier = torch.load(save_path_classifier)
    test_acc = test(online_network, classifier, loss_nll, test_dataloader, device)
    return test_acc

def main():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training with: {device}")

    os.makedirs('log_finetune', exist_ok=True)
    os.makedirs('model_weight', exist_ok=True)
    os.makedirs('test_result', exist_ok=True)
    os.makedirs('Base_fintune', exist_ok=True)

    test_acc_all = []

    for i in range(config['iteration']):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        writer = SummaryWriter(
            f"log_finetune/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_"
            f"FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot"
        )
        save_path_classifier = (
            f"model_weight/classifier_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_"
            f"FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        )
        save_path_online_network = (
            f"model_weight/online_network_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_"
            f"FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        )
        X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(
            config_ft['ft'], range(config_ft['class_start'], config_ft['class_end'] + 1),
            config_ft['k_shot'], seed=i
        )
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=i)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=config_ft['batch_size'], shuffle=True)
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=config_ft['batch_size'], shuffle=True)
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)

        online_network = torch.load('Base_trainer/epochend.pth')
        classifier = Classifier()
        loss_nll = nn.NLLLoss()
        online_network = online_network.to(device)
        classifier = classifier.to(device)
        loss_nll = loss_nll.to(device)

        optim_online_network = torch.optim.Adam(online_network.parameters(), lr=config_ft['lr'])
        optim_classifier = torch.optim.Adam(classifier.parameters(), lr=config_ft['lr'])
        scheduler_online_network = CosineAnnealingLR(optim_online_network, T_max=20)
        scheduler_classifier = CosineAnnealingLR(optim_classifier, T_max=20)

        test_acc = train_and_test(
            online_network, classifier, loss_nll, train_dataloader, val_dataloader, test_dataloader,
            optim_online_network, optim_classifier, scheduler_online_network, scheduler_classifier,
            config_ft['epochs'], save_path_online_network, save_path_classifier, device, writer
        )
        test_acc_all.append(test_acc)
        writer.close()

    test_acc_all_tensor = torch.tensor(test_acc_all)
    print(f"Sum of test accuracies: {test_acc_all_tensor.sum()}")
    print(f"Mean test accuracy: {test_acc_all_tensor.mean()}")
    df = pd.DataFrame(test_acc_all)
    df.to_excel(
        f"test_result/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_"
        f"FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot.xlsx"
    )

if __name__ == '__main__':
    main()