import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models import r3d_18, r2plus1d_18, ResCRNN


class CSL_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=500, train=True, transform=None):
        super(CSL_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.signers = 50
        self.repetition = 5
        if self.train:
            self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r',encoding='utf-8')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise

    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        start = 1
        step = int(len(os.listdir(folder_path))/self.frames)

        for i in range(self.frames):
            index = "{:06d}.jpg".format(start + i * step)
            image = Image.open(os.path.join(folder_path, index))  #.convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx/self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])
        if self.train:
            selected_folder = selected_folders[idx%self.videos_per_folder]
        else:
            selected_folder = selected_folders[idx%self.videos_per_folder + int(0.8*self.signers*self.repetition)]
        images = self.read_images(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.labels['{:06d}'.format(int(idx/self.videos_per_folder))])
        # label = self.labels['{:06d}'.format(int(idx/self.videos_per_folder))]
        label = torch.LongTensor([int(idx/self.videos_per_folder)])

        return {'data': images, 'label': label}

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]

def get_label_and_pred(model, dataloader, device):
    all_label = []
    all_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(pre_loader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            print(data['data'].shape)
            print("------------")
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute accuracy
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    all_label = all_label.squeeze().cpu().data.squeeze().numpy()
    all_pred = all_pred.cpu().data.squeeze().numpy()
    return all_label, all_pred


def plot_confusion_matrix(model, dataloader, device, save_path='confmat.png', normalize=True):
    # Get prediction
    all_label, all_pred = get_label_and_pred(model, dataloader, device)
    confmat = confusion_matrix(all_label, all_pred)

    # Normalize the matrix
    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    # Draw matrix
    plt.figure(figsize=(20,20))
    # confmat = np.random.rand(100,100)
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    # Add ticks
    ticks = np.arange(100)
    plt.xticks(ticks, fontsize=8)
    plt.yticks(ticks, fontsize=8)
    plt.grid(True)
    # Add title & labels
    plt.title('Confusion matrix', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    # Save figure
    plt.savefig(save_path)

    # Ranking
    sorted_index = np.diag(confmat).argsort()
    for i in range(100):
        # print(type(sorted_index[i]))
        print(pre_set.label_to_word(int(sorted_index[i])), confmat[sorted_index[i]][sorted_index[i]])
    # Save to csv
    np.savetxt('logs/confusion_matrix/'+model_name+'_matrix.csv', confmat, delimiter=',')


batch_size = 16
log_interval = 20
sample_size = 128
sample_duration = 16

num_classes = 100


data_path = r"C:\Sign\SLR_Dataset\CSL_Isolated\color_video_25000"
label_path = "../SLR_Dataset/CSL_Isolated/dictionary.txt"

model_path = "weight/LSTM_100/LSTM_epoch024.pth"
model_name = "LSTM_100"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    pre_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
                             num_classes=num_classes, train=False, transform=transform)
    pre_loader = DataLoader(pre_set, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    sample_size = 128
    sample_duration = 16  # 抽帧
    lstm_hidden_size = 512
    lstm_num_layers = 1
    attention = False
    model = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
                    lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, arch="resnet18",
                    attention=attention).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    plot_confusion_matrix(model, pre_loader, device, save_path='logs/confusion_matrix/'+model_name+'_confmat.png', normalize=True)