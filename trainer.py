import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import _create_model_training_folder, Data_augment_time, Data_augment_frequency, Data_to_frequency
import numpy as np
import matplotlib.pyplot as plt

class AAFALTrainer:
    def __init__(self, freq_online_network, freq_target_network, freq_predictor, freq_optimizer,
                 online_network, target_network, predictor, optimizer, device, **params):
        self.freq_online_network = freq_online_network
        self.freq_target_network = freq_target_network
        self.freq_optimizer = freq_optimizer
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.freq_predictor = freq_predictor
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.batch_size = params['batch_size']

        self.augment_time = Data_augment_time()
        self.augment_freq = Data_augment_frequency()
        self.to_frequency = Data_to_frequency()

        self.writer = SummaryWriter(
            f"runs/seed300_pretrain_class_in_{params['class_start']}-{params['class_end']}"
        )
        _create_model_training_folder(
            self.writer, files_to_same=["./config/config.yaml", "main.py", "trainer.py"]
        )

    @torch.no_grad()
    def _update_freq_target_network_parameters(self):
        for param_q, param_k in zip(self.freq_online_network.parameters(), self.freq_target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_freq_target_network(self):
        for param_q, param_k in zip(self.freq_online_network.parameters(), self.freq_target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def freq_train(self, train_loader, epoch_counter):
        self.freq_online_network.train()
        self.freq_predictor.train()
        train_loss_epoch = 0.0
        for batch_view, _ in train_loader:
            batch_view = batch_view.to(self.device)
            view1 = self.augment_time(batch_view)
            view2, freq_view2 = self.augment_freq(batch_view)
            freq_view1 = self.to_frequency(view1)
            self.freq_optimizer.zero_grad()
            loss = self.freq_update(freq_view1, freq_view2, view1, view2)
            loss.backward()
            self.freq_optimizer.step()
            train_loss_epoch += loss.item()
            self._update_freq_target_network_parameters()
        train_loss_epoch /= len(train_loader)
        self.writer.add_scalar('train_loss_epoch', train_loss_epoch, global_step=epoch_counter)
        print(f"The frequency loss on train dataset: {train_loss_epoch}")
        return train_loss_epoch

    def train(self, train_loader, epoch_counter):
        self.online_network.train()
        self.predictor.train()
        train_loss_epoch = 0.0
        for batch_view, _ in train_loader:
            batch_view = batch_view.to(self.device)
            view1 = self.augment_time(batch_view)
            view2, freq_view2 = self.augment_freq(batch_view)
            freq_view1 = self.to_frequency(view1)
            self.optimizer.zero_grad()
            loss = self.update(freq_view1, freq_view2, view1, view2)
            loss.backward()
            self.optimizer.step()
            train_loss_epoch += loss.item()
            self._update_target_network_parameters()
        train_loss_epoch /= len(train_loader)
        self.writer.add_scalar('train_loss_epoch', train_loss_epoch, global_step=epoch_counter)
        print(f"The loss on train dataset: {train_loss_epoch}")
        return train_loss_epoch

    def eval(self, val_loader, epoch_counter):
        self.online_network.eval()
        self.predictor.eval()
        eval_loss_epoch = 0.0
        for batch_view, _ in val_loader:
            batch_view = batch_view.to(self.device)
            view1 = self.augment_time(batch_view)
            view2, _ = self.augment_freq(batch_view)
            with torch.no_grad():
                loss = self.eval_update(view1, view2)
                eval_loss_epoch += loss.item()
        eval_loss_epoch /= len(val_loader)
        self.writer.add_scalar('eval_loss_epoch', eval_loss_epoch, global_step=epoch_counter)
        print(f"The loss on eval dataset: {eval_loss_epoch}")
        return eval_loss_epoch

    def train_and_val(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  drop_last=False, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                drop_last=False, shuffle=True)

        self.initializes_freq_target_network()
        self.initializes_target_network()

        loss_min = float('inf')
        freq_train_lossP, train_lossP, eval_lossP = [], [], []

        for epoch_counter in range(self.max_epochs):
            print(f'Epoch={epoch_counter}')
            freq_train_loss_epoch = self.freq_train(train_loader, epoch_counter)
            train_loss_epoch = self.train(train_loader, epoch_counter)
            eval_loss_epoch = self.eval(val_loader, epoch_counter)

            freq_train_lossP.append(freq_train_loss_epoch)
            train_lossP.append(train_loss_epoch)
            eval_lossP.append(eval_loss_epoch)

            if eval_loss_epoch <= loss_min:
                torch.save(self.online_network.state_dict(), 'Base_trainer/epoch_best.pth')
                loss_min = eval_loss_epoch

            if epoch_counter % 50 == 0:
                torch.save(self.online_network.state_dict(), f'Base_trainer/epoch{epoch_counter}.pth')
                torch.save(self.freq_online_network.state_dict(), f'Base_trainer/epoch{epoch_counter}_freq.pth')

            print("End of epoch {}".format(epoch_counter))

        torch.save(self.online_network.state_dict(), 'Base_trainer/epochend.pth')
        torch.save(self.freq_online_network.state_dict(), 'Base_trainer/epochend_freq.pth')

        plt.figure()
        plt.plot(range(len(freq_train_lossP)), freq_train_lossP, linewidth=1, linestyle="solid", label="freq train loss")
        plt.plot(range(len(train_lossP)), train_lossP, linewidth=1, linestyle="solid", label="train loss")
        plt.plot(range(len(eval_lossP)), eval_lossP, linewidth=1, linestyle="solid", label="eval loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Loss curve')
        plt.savefig('Base_trainer/loss.png')
        plt.close()
        np.savetxt('Base_trainer/train_freq_loss.txt', freq_train_lossP)
        np.savetxt('Base_trainer/train_loss.txt', train_lossP)
        np.savetxt('Base_trainer/eval_loss.txt', eval_lossP)

    def freq_update(self, freq_view1, freq_view2, view1, view2):
        pred1 = self.predictor(self.online_network(view1)[1])
        pred2 = self.predictor(self.online_network(view2)[1])
        freq_pred1 = self.freq_predictor(self.freq_online_network(freq_view1)[1])
        freq_pred2 = self.freq_predictor(self.freq_online_network(freq_view2)[1])

        with torch.no_grad():
            freq_target1 = self.freq_target_network(freq_view2)[1]
            freq_target2 = self.freq_target_network(freq_view1)[1]

        loss = self.regression_loss(freq_pred1, freq_target1)
        loss += self.regression_loss(freq_pred2, freq_target2)
        loss += self.regression_loss(pred1, freq_pred1)
        loss += self.regression_loss(pred2, freq_pred2)
        return loss.mean()

    def update(self, freq_view1, freq_view2, view1, view2):
        pred1 = self.predictor(self.online_network(view1)[1])
        pred2 = self.predictor(self.online_network(view2)[1])
        freq_pred1 = self.freq_predictor(self.freq_online_network(freq_view1)[1])
        freq_pred2 = self.freq_predictor(self.freq_online_network(freq_view2)[1])

        with torch.no_grad():
            target1 = self.target_network(view2)[1]
            target2 = self.target_network(view1)[1]

        loss = self.regression_loss(pred1, target1)
        loss += self.regression_loss(pred2, target2)
        loss += self.regression_loss(pred1, freq_pred1)
        loss += self.regression_loss(pred2, freq_pred2)
        return loss.mean()

    def eval_update(self, view1, view2):
        pred1 = self.predictor(self.online_network(view1)[1])
        pred2 = self.predictor(self.online_network(view2)[1])

        with torch.no_grad():
            target1 = self.target_network(view2)[1]
            target2 = self.target_network(view1)[1]

        loss = self.regression_loss(pred1, target1)
        loss += self.regression_loss(pred2, target2)
        return loss.mean()
