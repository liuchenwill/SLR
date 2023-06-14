
import streamlit as st
import torch
from models import r2plus1d_18, r3d_18, ResCRNN
import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor

def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),

    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize([128, 128]),
    ])
    return transform


def play_webcam(selected_model, selected_weight):
    SOFTMAX_THRES = st.sidebar.slider('SOFTMAX_THRES', 0.0, 1.0, 0.75)
    print(SOFTMAX_THRES)
    if st.sidebar.button('开始'):
        try:
            labels = []
            label_path = '../SLR_Dataset/CSL_Isolated/dictionary.txt'
            try:
                label_file = open(label_path, 'r', encoding='utf-8')
                for line in label_file.readlines():
                    line = line.strip()
                    line = line.split('\t')
                    labels.append(line[1])
            except Exception as e:
                raise

            # 模型选择
            num_classes = int(selected_model[-3:])
            if (selected_model == "r2+1d_100") | (selected_model == "r2+1d_500"):
                model_path = "weight/" + str(selected_model) + "/" + str(selected_weight)
                model = r2plus1d_18(num_classes=num_classes).cuda()
                model.load_state_dict(torch.load(model_path))
                model.eval()

            elif (selected_model == "r3d_100") | (selected_model == "r3d_500"):
                model_path = "weight/" + str(selected_model) + "/" + str(selected_weight)
                model = r3d_18(num_classes=num_classes).cuda()
                model.load_state_dict(torch.load(model_path))
                model.eval()

            elif (selected_model == "LSTM_100") | (selected_model == "LSTM_500"):
                model_path = "weight/" + str(selected_model) + "/" + str(selected_weight)
                sample_size = 128
                sample_duration = 16  # 抽帧
                lstm_hidden_size = 512
                lstm_num_layers = 1
                attention = False
                model = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
                                lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, arch="resnet18",
                                attention=attention).cuda()
                model.load_state_dict(torch.load(model_path))
                model.eval()


            cap = cv2.VideoCapture(0)

            # set a lower resolution for speed up   为加速设置一个较低的分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


            index = 0
            print("Build transformer...")
            transform = get_transform()  # 预处理
            print("Build Executor...")
            idx = 0
            _idx = 0
            cnt = 0
            i_frame = -1

            print("Ready!")
            history_logit = [i - i for i in range(100)]

            cache_input = torch.ones(1, 3, 16, 128, 128).cuda()

            st1, st2, st3 = st.columns(3)
            with st1:
                st.markdown("### 预测词汇")
                st1_text = st.markdown(f"NULL")
            with st2:
                st.markdown("### Confidence Level")
                st2_text = st.markdown(f"{0}")
            with st3:
                st.markdown("### FPS")
                st3_text = st.markdown(f"{0}")
            # st.markdown("---")
            st_frame = st.empty()
            while (cap.isOpened()):
                success, image = cap.read()
                i_frame += 1
                if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate ， 隔帧抽取
                    t1 = time.time()
                    img_tran = transform([Image.fromarray(image).convert('RGB')])  # 图片预处理
                    cache_input = cache_input[:, :, 1:, :, :]
                        # print(cache_input.shape)
                    input_var = torch.autograd.Variable(
                        img_tran.view(1, 3, 1, img_tran.size(1), img_tran.size(2))).cuda()  # 张量转换
                    cache_input = torch.cat((cache_input, input_var), dim=2)
                        # print(cache_input.shape)
                    with torch.no_grad():
                        feat = model(cache_input)

                    if SOFTMAX_THRES > 0:
                        feat = feat.cpu()
                        feat_np = feat.numpy().reshape(-1)
                        feat_np -= feat_np.max()
                        softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                        # print(max(softmax))
                        if max(softmax) > SOFTMAX_THRES:
                            idx = np.argmax(feat.numpy(), axis=1)[0]
                        else:
                            _idx = np.argmax(feat.numpy(), axis=1)[0]
                        # print(idx)


                    t2 = time.time()
                    # print(f"{index} {labels[_idx]}")
                    index += 1
                    current_time = t2 - t1  # 推理时间


                if success:
                    # image = cv2.resize(image, (640, 480))
                    st_frame.image(image, caption='Real Video', channels="BGR", use_column_width=True)
                    if(max(softmax)>SOFTMAX_THRES):
                        cnt = 0

                    if cnt<40:
                        st1_text.markdown(f"  **{labels[idx]}**  ")
                        cnt+=1
                    else:
                        st1_text.markdown(f"  **NULL**  ")

                    st2_text.markdown("{}: {:.2f} %".format(labels[_idx], max(softmax)*100))
                    st3_text.markdown('{:.1f} Vid/s'.format(1 / current_time))
                else:
                    cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

