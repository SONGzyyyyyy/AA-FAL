import os
from shutil import copyfile
import torch
import torch.nn as nn

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

class Data_to_frequency(nn.Module):
    def __init__(self, time_to_freq=False):
        super(Data_to_frequency, self).__init__()
        self.time_to_freq = time_to_freq

    def Time_to_Freq(self, x):
        fft_result = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(fft_result)
        return magnitude

    def forward(self, x):
        x = x.cuda()
        x = self.Time_to_Freq(x)
        max_value = torch.max(x)
        min_value = torch.min(x)
        x = (x - min_value) / (max_value - min_value)
        return x.detach()

class Data_augment_time(nn.Module):
    def __init__(self):
        super(Data_augment_time, self).__init__()
        self.time_len_per = 1200
        self.time_scale_per = 10
        self.time_scale_times = 1
        self.time_scale_plus = 1
        self.time_len = 1200

    def Time_Perturbation_Stop(self, x):
        aug_time_select = torch.randint(0, 4, [1])
        if aug_time_select == 0:
            d = torch.randn_like(x) + 0.5
            x_slice = self.time_scale_per * d
            start = torch.randint(0, x.shape[2] - self.time_len_per, [1])
            end = start + self.time_len_per
            x_slice[:, 0, 0:start] = x[:, 0, 0:start]
            x_slice[:, 0, end:] = x[:, 0, end:]
            x_slice[:, 1, :] = x[:, 1, :]
        elif aug_time_select == 1:
            d = torch.randn_like(x) + 0.5
            x_slice = self.time_scale_per * d
            start = torch.randint(0, x.shape[2] - self.time_len_per, [1])
            end = start + self.time_len_per
            x_slice[:, 1, 0:start] = x[:, 1, 0:start]
            x_slice[:, 1, end:] = x[:, 1, end:]
            x_slice[:, 0, :] = x[:, 0, :]
        elif aug_time_select == 2:
            x_slice = torch.zeros_like(x) + 0.5
            d = self.time_scale_per * torch.randn_like(x[:, 0, :])
            x_slice[:, 0, :] = d
            x_slice[:, 1, :] = d
            start = torch.randint(0, x.shape[2] - self.time_len_per, [1])
            end = start + self.time_len_per
            x_slice[:, :, 0:start] = x[:, :, 0:start]
            x_slice[:, :, end:] = x[:, :, end:]
        elif aug_time_select == 3:
            d = torch.randn_like(x) + 0.5
            x_slice = self.time_scale_per * d
            start_0 = torch.randint(0, x.shape[2] - self.time_len_per, [1])
            end_0 = start_0 + self.time_len_per
            x_slice[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_slice[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.time_len_per, [1])
            end_1 = start_1 + self.time_len_per
            x_slice[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_slice[:, 1, end_1:] = x[:, 1, end_1:]
        return x_slice

    def Time_local_Times(self, x, mean=1, sigma=0.1):
        aug_time_select = torch.randint(0, 4, [1])
        if aug_time_select == 0:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.time_scale_times * d
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_jitter[:, 0, 0:start] = x[:, 0, 0:start]
            x_jitter[:, 0, end:] = x[:, 0, end:]
            x_jitter[:, 1, :] = x[:, 1, :]
        elif aug_time_select == 1:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.time_scale_times * d
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_jitter[:, 1, 0:start] = x[:, 1, 0:start]
            x_jitter[:, 1, end:] = x[:, 1, end:]
            x_jitter[:, 0, :] = x[:, 0, :]
        elif aug_time_select == 2:
            x_jitter = torch.zeros_like(x)
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter[:, 0, :] = self.time_scale_times * d[:, 0, :]
            x_jitter[:, 1, :] = self.time_scale_times * d[:, 0, :]
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_jitter[:, :, 0:start] = x[:, :, 0:start]
            x_jitter[:, :, end:] = x[:, :, end:]
        elif aug_time_select == 3:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.time_scale_times * d
            start_0 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_0 = start_0 + self.time_len
            x_jitter[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_jitter[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_1 = start_1 + self.time_len
            x_jitter[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_jitter[:, 1, end_1:] = x[:, 1, end_1:]
        return x_jitter

    def Time_local_Times_Constant(self, x, mean=1, sigma=0.1):
        aug_time_select = torch.randint(0, 4, [1])
        if aug_time_select == 0:
            scalingFactor = self.time_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                 device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x * scalingTensor
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_scaling[:, 0, 0:start] = x[:, 0, 0:start]
            x_scaling[:, 0, end:] = x[:, 0, end:]
            x_scaling[:, 1, :] = x[:, 1, :]
        elif aug_time_select == 1:
            scalingFactor = self.time_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                 device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x * scalingTensor
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_scaling[:, 1, 0:start] = x[:, 1, 0:start]
            x_scaling[:, 1, end:] = x[:, 1, end:]
            x_scaling[:, 0, :] = x[:, 0, :]
        elif aug_time_select == 2:
            scalingFactor = self.time_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                 device=x.device)
            scalingFactor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            scalingTensor = torch.zeros_like(scalingFactor)
            scalingTensor[:, 0, :] = scalingFactor[:, 0, :]
            scalingTensor[:, 1, :] = scalingFactor[:, 0, :]
            x_scaling = x * scalingTensor
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_scaling[:, :, 0:start] = x[:, :, 0:start]
            x_scaling[:, :, end:] = x[:, :, end:]
        elif aug_time_select == 3:
            scalingFactor = self.time_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                 device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x * scalingTensor
            start_0 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_0 = start_0 + self.time_len
            x_scaling[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_scaling[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_1 = start_1 + self.time_len
            x_scaling[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_scaling[:, 1, end_1:] = x[:, 1, end_1:]
        return x_scaling

    def Time_local_Plus(self, x, mean=0, sigma=1):
        aug_time_select = torch.randint(0, 4, [1])
        if aug_time_select == 0:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.time_scale_plus * d
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_jitter[:, 0, 0:start] = x[:, 0, 0:start]
            x_jitter[:, 0, end:] = x[:, 0, end:]
            x_jitter[:, 1, :] = x[:, 1, :]
        elif aug_time_select == 1:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.time_scale_plus * d
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_jitter[:, 1, 0:start] = x[:, 1, 0:start]
            x_jitter[:, 1, end:] = x[:, 1, end:]
            x_jitter[:, 0, :] = x[:, 0, :]
        elif aug_time_select == 2:
            x_jitter = torch.zeros_like(x)
            d = self.time_scale_plus * torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter[:, 0, :] = d[:, 0, :]
            x_jitter[:, 1, :] = d[:, 0, :]
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_jitter[:, :, 0:start] = x[:, :, 0:start]
            x_jitter[:, :, end:] = x[:, :, end:]
        elif aug_time_select == 3:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.time_scale_plus * d
            start_0 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_0 = start_0 + self.time_len
            x_jitter[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_jitter[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_1 = start_1 + self.time_len
            x_jitter[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_jitter[:, 1, end_1:] = x[:, 1, end_1:]
        return x_jitter

    def Time_local_Plus_Constant(self, x, mean=0, sigma=1):
        aug_time_select = torch.randint(0, 4, [1])
        if aug_time_select == 0:
            scalingFactor = self.time_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x + scalingTensor
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_scaling[:, 0, 0:start] = x[:, 0, 0:start]
            x_scaling[:, 0, end:] = x[:, 0, end:]
            x_scaling[:, 1, :] = x[:, 1, :]
        elif aug_time_select == 1:
            scalingFactor = self.time_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x + scalingTensor
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_scaling[:, 1, 0:start] = x[:, 1, 0:start]
            x_scaling[:, 1, end:] = x[:, 1, end:]
            x_scaling[:, 0, :] = x[:, 0, :]
        elif aug_time_select == 2:
            scalingFactor = self.time_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                device=x.device)
            scalingFactor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            scalingTensor = torch.zeros_like(scalingFactor)
            scalingTensor[:, 0, :] = scalingFactor[:, 0, :]
            scalingTensor[:, 1, :] = scalingFactor[:, 0, :]
            x_scaling = x + scalingTensor
            start = torch.randint(0, x.shape[2] - self.time_len, [1])
            end = start + self.time_len
            x_scaling[:, :, 0:start] = x[:, :, 0:start]
            x_scaling[:, :, end:] = x[:, :, end:]
        elif aug_time_select == 3:
            scalingFactor = self.time_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x + scalingTensor
            start_0 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_0 = start_0 + self.time_len
            x_scaling[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_scaling[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.time_len, [1])
            end_1 = start_1 + self.time_len
            x_scaling[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_scaling[:, 1, end_1:] = x[:, 1, end_1:]
        return x_scaling

    def forward(self, x):
        x = x.cuda()
        aug_time_select = torch.randint(0, 16, [1])
        if aug_time_select == 0:
            x = self.Time_local_Times_Constant(x)
        elif aug_time_select == 1:
            x = self.Time_local_Times(x)
        elif aug_time_select in [2, 3]:
            x = self.Time_local_Plus_Constant(x)
        elif aug_time_select in [4, 5, 6, 7]:
            x = self.Time_local_Plus(x)
        x = self.Time_Perturbation_Stop(x)
        return x.detach()

class Data_augment_frequency(nn.Module):
    def __init__(self):
        super(Data_augment_frequency, self).__init__()
        self.frequency_len = 1200
        self.frequency_len_per = 1200
        self.frequency_scale_per = 0.1
        self.frequency_scale_times = 1
        self.frequency_scale_plus = 0.01

    def Frequency_Stop_Perturbation(self, x):
        aug_fft_select = torch.randint(0, 4, [1])
        if aug_fft_select == 0:
            d = torch.randn_like(x)
            x_slice = self.frequency_scale_per * d
            start = torch.randint(0, x.shape[2] - self.frequency_len_per, [1])
            end = start + self.frequency_len_per
            x_slice[:, 0, 0:start] = x[:, 0, 0:start]
            x_slice[:, 0, end:] = x[:, 0, end:]
            x_slice[:, 1, :] = x[:, 1, :]
        elif aug_fft_select == 1:
            d = torch.randn_like(x)
            x_slice = self.frequency_scale_per * d
            start = torch.randint(0, x.shape[2] - self.frequency_len_per, [1])
            end = start + self.frequency_len_per
            x_slice[:, 1, 0:start] = x[:, 1, 0:start]
            x_slice[:, 1, end:] = x[:, 1, end:]
            x_slice[:, 0, :] = x[:, 0, :]
        elif aug_fft_select == 2:
            x_slice = torch.zeros_like(x)
            d = self.frequency_scale_per * torch.randn_like(x[:, 0, :])
            x_slice[:, 0, :] = d
            x_slice[:, 1, :] = d
            start = torch.randint(0, x.shape[2] - self.frequency_len_per, [1])
            end = start + self.frequency_len_per
            x_slice[:, :, 0:start] = x[:, :, 0:start]
            x_slice[:, :, end:] = x[:, :, end:]
        elif aug_fft_select == 3:
            d = torch.randn_like(x)
            x_slice = self.frequency_scale_per * d
            start_0 = torch.randint(0, x.shape[2] - self.frequency_len_per, [1])
            end_0 = start_0 + self.frequency_len_per
            x_slice[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_slice[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.frequency_len_per, [1])
            end_1 = start_1 + self.frequency_len_per
            x_slice[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_slice[:, 1, end_1:] = x[:, 1, end_1:]
        return x_slice

    def Frequency_local_Times(self, x, mean=1, sigma=0.01):
        aug_fft_select = torch.randint(0, 4, [1])
        if aug_fft_select == 0:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.frequency_scale_times * d
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_jitter[:, 0, 0:start] = x[:, 0, 0:start]
            x_jitter[:, 0, end:] = x[:, 0, end:]
            x_jitter[:, 1, :] = x[:, 1, :]
        elif aug_fft_select == 1:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.frequency_scale_times * d
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_jitter[:, 1, 0:start] = x[:, 1, 0:start]
            x_jitter[:, 1, end:] = x[:, 1, end:]
            x_jitter[:, 0, :] = x[:, 0, :]
        elif aug_fft_select == 2:
            x_jitter = torch.zeros_like(x)
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter[:, 0, :] = self.frequency_scale_times * d[:, 0, :]
            x_jitter[:, 1, :] = self.frequency_scale_times * d[:, 0, :]
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_jitter[:, :, 0:start] = x[:, :, 0:start]
            x_jitter[:, :, end:] = x[:, :, end:]
        elif aug_fft_select == 3:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.frequency_scale_times * d
            start_0 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_0 = start_0 + self.frequency_len
            x_jitter[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_jitter[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_1 = start_1 + self.frequency_len
            x_jitter[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_jitter[:, 1, end_1:] = x[:, 1, end_1:]
        return x_jitter

    def Frequency_local_Times_Constant(self, x, mean=1, sigma=0.01):
        aug_fft_select = torch.randint(0, 4, [1])
        if aug_fft_select == 0:
            scalingFactor = self.frequency_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                      device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x * scalingTensor
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_scaling[:, 0, 0:start] = x[:, 0, 0:start]
            x_scaling[:, 0, end:] = x[:, 0, end:]
            x_scaling[:, 1, :] = x[:, 1, :]
        elif aug_fft_select == 1:
            scalingFactor = self.frequency_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                      device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x * scalingTensor
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_scaling[:, 1, 0:start] = x[:, 1, 0:start]
            x_scaling[:, 1, end:] = x[:, 1, end:]
            x_scaling[:, 0, :] = x[:, 0, :]
        elif aug_fft_select == 2:
            scalingFactor = self.frequency_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                      device=x.device)
            scalingFactor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            scalingTensor = torch.zeros_like(scalingFactor)
            scalingTensor[:, 0, :] = scalingFactor[:, 0, :]
            scalingTensor[:, 1, :] = scalingFactor[:, 0, :]
            x_scaling = x * scalingTensor
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_scaling[:, :, 0:start] = x[:, :, 0:start]
            x_scaling[:, :, end:] = x[:, :, end:]
        elif aug_fft_select == 3:
            scalingFactor = self.frequency_scale_times * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                      device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x * scalingTensor
            start_0 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_0 = start_0 + self.frequency_len
            x_scaling[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_scaling[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_1 = start_1 + self.frequency_len
            x_scaling[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_scaling[:, 1, end_1:] = x[:, 1, end_1:]
        return x_scaling

    def Frequency_local_Plus(self, x, mean=0, sigma=1):
        aug_fft_select = torch.randint(0, 4, [1])
        if aug_fft_select == 0:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.frequency_scale_plus * d
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_jitter[:, 0, 0:start] = x[:, 0, 0:start]
            x_jitter[:, 0, end:] = x[:, 0, end:]
            x_jitter[:, 1, :] = x[:, 1, :]
        elif aug_fft_select == 1:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.frequency_scale_plus * d
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_jitter[:, 1, 0:start] = x[:, 1, 0:start]
            x_jitter[:, 1, end:] = x[:, 1, end:]
            x_jitter[:, 0, :] = x[:, 0, :]
        elif aug_fft_select == 2:
            x_jitter = torch.zeros_like(x)
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter[:, 0, :] = d[:, 0, :]
            x_jitter[:, 1, :] = d[:, 0, :]
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_jitter[:, :, 0:start] = x[:, :, 0:start]
            x_jitter[:, :, end:] = x[:, :, end:]
        elif aug_fft_select == 3:
            d = torch.normal(mean=mean, std=sigma, size=x.shape, device=x.device)
            x_jitter = self.frequency_scale_plus * d
            start_0 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_0 = start_0 + self.frequency_len
            x_jitter[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_jitter[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_1 = start_1 + self.frequency_len
            x_jitter[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_jitter[:, 1, end_1:] = x[:, 1, end_1:]
        return x_jitter

    def Frequency_local_Plus_Constant(self, x, mean=0, sigma=1):
        aug_fft_select = torch.randint(0, 4, [1])
        if aug_fft_select == 0:
            scalingFactor = self.frequency_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                     device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x + scalingTensor
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_scaling[:, 0, 0:start] = x[:, 0, 0:start]
            x_scaling[:, 0, end:] = x[:, 0, end:]
            x_scaling[:, 1, :] = x[:, 1, :]
        elif aug_fft_select == 1:
            scalingFactor = self.frequency_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                     device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x + scalingTensor
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_scaling[:, 1, 0:start] = x[:, 1, 0:start]
            x_scaling[:, 1, end:] = x[:, 1, end:]
            x_scaling[:, 0, :] = x[:, 0, :]
        elif aug_fft_select == 2:
            scalingFactor = self.frequency_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                     device=x.device)
            scalingFactor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            scalingTensor = torch.zeros_like(scalingFactor)
            scalingTensor[:, 0, :] = scalingFactor[:, 0, :]
            scalingTensor[:, 1, :] = scalingFactor[:, 0, :]
            x_scaling = x + scalingTensor
            start = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end = start + self.frequency_len
            x_scaling[:, :, 0:start] = x[:, :, 0:start]
            x_scaling[:, :, end:] = x[:, :, end:]
        elif aug_fft_select == 3:
            scalingFactor = self.frequency_scale_plus * torch.normal(mean=mean, std=sigma, size=(x.shape[0], x.shape[1]),
                                                                     device=x.device)
            scalingTensor = scalingFactor.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x_scaling = x + scalingTensor
            start_0 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_0 = start_0 + self.frequency_len
            x_scaling[:, 0, 0:start_0] = x[:, 0, 0:start_0]
            x_scaling[:, 0, end_0:] = x[:, 0, end_0:]
            start_1 = torch.randint(0, x.shape[2] - self.frequency_len, [1])
            end_1 = start_1 + self.frequency_len
            x_scaling[:, 1, 0:start_1] = x[:, 1, 0:start_1]
            x_scaling[:, 1, end_1:] = x[:, 1, end_1:]
        return x_scaling

    def forward(self, x):
        x = x.cuda()
        fft_result = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(fft_result)
        max_value = torch.max(magnitude)
        min_value = torch.min(magnitude)
        magnitude = (magnitude - min_value) / (max_value - min_value)
        aug_frequency_select = torch.randint(0, 16, [1])
        if aug_frequency_select == 0:
            fft_result = self.Frequency_local_Times_Constant(fft_result)
        elif aug_frequency_select == 1:
            fft_result = self.Frequency_local_Times(fft_result)
        elif aug_frequency_select in [2, 3]:
            fft_result = self.Frequency_local_Plus_Constant(fft_result)
        elif aug_frequency_select in [4, 5, 6, 7]:
            fft_result = self.Frequency_local_Plus(fft_result)
        fft_result = self.Frequency_Stop_Perturbation(fft_result)
        x = torch.fft.ifft(fft_result).real
        max_value = torch.max(x)
        min_value = torch.min(x)
        x = (x - min_value) / (max_value - min_value)
        return x.detach(), magnitude