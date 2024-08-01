import torch
import os
import numpy as np
from models import Deepfas_Inclusion_CoAtNet
import cv2
import torch.nn.functional as F



def np_normalize(a, print_flag=False):
    amin, amax = a.min(), a.max() # 求最大最小值
    if print_flag: print('Before normalize: ', amin, amax)
    b = (a-amin+1e-7) / (amax-amin+1e-7) # (矩阵元素-最小值)/(最大值-最小值)
    if print_flag: print('After normalize: ', b.min(), b.max())
    return b


def generate_DCT_3C(image_bgr, size=None, color_type="YCrCb"):
    assert color_type in ["GRAY", "YUV", "RGB", "BGR", "YCrCb"]
    image = cv2.cvtColor(image_bgr, eval("cv2.COLOR_BGR2{}".format(color_type)))
    if size is not None: image = cv2.resize(image, size)
    image = image.astype(np.float)
    img_dct = np.zeros(shape=size[::-1]+(3,), dtype=np.float64)
    img_dct[..., 0] = np_normalize(np.log(abs(cv2.dct(image[..., 0]))+1e-7))
    img_dct[..., 1] = np_normalize(np.log(abs(cv2.dct(image[..., 1]))+1e-7))
    img_dct[..., 2] = np_normalize(np.log(abs(cv2.dct(image[..., 2]))+1e-7))
    return img_dct


def load_model_weights(model, weights_path=None, strict=True):

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['state_dict']
    if "model" in state_dict: 
        state_dict = state_dict["model"]

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    own_state = model.state_dict()
    for name, param in state_dict.items():
        # try to get correct name
        if name in own_state: pass
        elif name.replace('module.','',1) in own_state:
            name = name.replace('module.','',1)  # rename, delete 'module.'
        elif 'module.'+name in own_state:
            name = 'module.'+name  # rename, add 'module.'

        if name in own_state:
            if param.shape == own_state[name].shape:
                new_state_dict[name] = param
            else:
                mycfg.debug_print("Skip for mistaken shape: {}, {}".format(name, param.shape))
        else:
            mycfg.debug_print("Skip for mistaken name: {}, {}".format(name, param.shape))
    model.load_state_dict(new_state_dict, strict=strict)
    return model


model_path = './classifier/for_full_exp_general/inclusion_coatnet/Jul02_09-33-49/17_acc@1_0.701255_tpr@5e-3_0.257292_thresh@5e-3_0.993402.pth.tar'

model = Deepfas_Inclusion_CoAtNet()

model = load_model_weights(model, model_path)

img = cv2.imread('./imgs/7ab02a23cf8a30cda4c70647b6c34e4b.jpg')

img_fft1 = np.zeros((3,224,224))
img_dct1 = np.transpose(generate_DCT_3C(img, (224,224), color_type="YCrCb"), (2, 0, 1))

mean = np.array([0.485, 0.456, 0.406], dtype=np.float).reshape((1,1,3))
std = np.array([0.229, 0.224, 0.225], dtype=np.float).reshape((1,1,3))
img = cv2.resize(img, (224,224))[:,:,::-1]
img = np.transpose((img / 255. - mean) / std, (2,0,1))

img_inputs = np.concatenate([img, img_fft1, img_dct1], axis=0)
img_inputs = np.expand_dims(img_inputs, axis=0).astype("float32")

img_inputs = torch.tensor(img_inputs)

model = model.eval()

output = model(img_inputs, phase="only_prediction")
prob = output[0].ravel()
prob = F.softmax(prob)[1]
print(prob)
