import os

import torch
from models import r2plus1d_18
import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time


SOFTMAX_THRES = 0.7
HISTORY_LOGIT = True

REFINE_OUTPUT = True
history_logit = []

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


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (128, 128))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame

history=[0]


n_still_frame = 0

WINDOW_NAME = 'Sign Language Recognition'

def real_pre():
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


    torch_module = r2plus1d_18(num_classes=100).cuda()
    torch_module.load_state_dict(torch.load("weight/r2+1d_100/r2+1d18_epoch012.pth"))
    torch_module.eval()

    print("Open camera...")
    cap = cv2.VideoCapture(0)  # 打开摄像头

    print(cap)

    # set a lower resolution for speed up   为加速设置一个较低的分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables  窗口变量
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    transform = get_transform()  # 预处理
    print("Build Executor...")

    idx = 0
    i_frame = -1

    print("Ready!")
    history_logit = [i-i for i in range(100)]
    cnt = 0
    cache_input = torch.ones(1, 3, 16, 128, 128).cuda()
    while True:  # 读取摄像头
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate ， 隔帧抽取
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])  # 图片预处理
            cache_input = cache_input[:, :, 1:, :, :]
            # print(cache_input.shape)
            input_var = torch.autograd.Variable(img_tran.view(1, 3, 1,img_tran.size(1), img_tran.size(2))) .cuda() # 张量转换
            cache_input = torch.cat((cache_input, input_var), dim=2)
            # print(cache_input.shape)
            with torch.no_grad():
                feat = torch_module(cache_input)


            if SOFTMAX_THRES > 0:
                feat = feat.cpu()
                feat_np = feat.numpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx = np.argmax(feat.numpy(), axis=1)[0]
                print(idx)


            # if HISTORY_LOGIT:  # 平均
            #     cnt += 1
            #     if cnt % 10 == 0:
            #         history_logit += feat.numpy()
            #         avg_logit = sum(history_logit)
            #
            #     idx = np.argmax(avg_logit, axis=1)[0]
            #     print(idx)



            t2 = time.time()
            print(f"{index} {labels[idx]}")

            current_time = t2 - t1  # 推理时间

        # 识别效果展示部分
        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + str(idx),
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


real_pre()

