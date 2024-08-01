from numpy.lib.arraysetops import isin
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np, pandas as pd
import cv2, json, random
from copy import deepcopy
from collections import Counter
from imgaug import augmenters as iaa

import dataset_transforms


def dataset_check(data_path='./data/train_label.txt', dataset_dir='../../../../'):
    '''删除datalist中 不存在或fsize=0 的img_path'''

    root_df = pd.read_csv(data_path, sep=' ', header=None, names=['img_path', 'target'])
    root_df['img_path'] = root_df['img_path'].astype(str)
    root_df['target']   = root_df['target'].fillna(0).astype(int)  # target这一列并不一定存在，缺省时自动补为0
    old_len = root_df.shape[0]
    print('Before dataset_check, old_len: ', old_len)

    root_df.img_path    = root_df.img_path.apply(lambda x: os.path.join(dataset_dir, x))
    root_df.drop(root_df[root_df.img_path.apply(lambda x: (not os.path.exists(x)))].index, inplace=True)  # file exists
    root_df.drop(root_df[root_df.img_path.apply(lambda x: (os.path.getsize(x)==0))].index, inplace=True)  # file not empty
    new_len = root_df.shape[0]
    print('After  dataset_check, new_len: ', new_len)

    # dataset_processing.py: calling main function ...
    # Before dataset_check, old_len:  275662
    # After  dataset_check, new_len:  275662

def load_dataset_csv(data_path='', data_ratio=1.0, sep=','):
    try: 
        df = pd.read_csv(data_path, sep=sep)[['img_path', 'target']]
    except:
        df = pd.read_csv(data_path, sep=sep, header=None, names={'img_path':str, 'target':int})
    df = df.sample(frac=data_ratio).reset_index(drop=True)
    return df

def split_neg_pos_datadf(data_path='', data_ratio=1.0):
    if isinstance(data_path, list):
        data_df = pd.concat([load_dataset_csv(p, data_ratio) for p in data_path], axis=0)
    elif isinstance(data_path, dict):
        data_df = pd.concat([load_dataset_csv(p, data_ratio*r) for p,r in data_path.items()], axis=0)
    elif isinstance(data_path, str):
        data_df = load_dataset_csv(data_path, data_ratio)
    else:
        print(f'Check error data_path:{data_path}')
    df_white = data_df[ data_df['target']==0 ].reset_index(drop=True)
    df_black = data_df[ data_df['target']==1 ].reset_index(drop=True)
    
    num1 = df_white.shape[0]
    num2 = df_black.shape[0]
    if num1>num2:
        num_gap = num1-num2
        df_temp = df_black.sample(frac=num_gap/num2).reset_index(drop=True)
        df_black = pd.concat([df_black, df_temp], axis=0) 
    elif num1<num2:
        num_gap = num2-num1
        df_temp = df_white.sample(frac=num_gap/num2).reset_index(drop=True)
        df_white = pd.concat([df_white, df_temp], axis=0) 

    num_final = min(df_white.shape[0], df_black.shape[0])
    df_white = df_white.iloc[:num_final]
    df_black = df_black.iloc[:num_final]
    assert df_white.shape[0]==df_black.shape[0]
    return df_white, df_black

class DatasetProcessing(Dataset):
    def __init__(self, data_path='', phase='Train', transform=None, data_ratio=1.0, data_df=None, shuffle=True, data_dir='', load_func=None):
        self.transform = transform
        self.phase = phase
        self.data_dir = data_dir
        
        self.with_mask = False  # general_df, close mask
        self.mask_transform = dataset_transforms.get_mask_transform(img_size=(224,224))

        print('='*20+f' Start to load dataset...')

        load_func = deepcopy(self.load_dataset_csv) if load_func is None else load_func
        if data_df is None:
            if isinstance(data_path, list) or isinstance(data_path, set):
                print('Use data_path list/set')
                data_df = pd.concat([load_func(p, data_ratio) for p in data_path], axis=0)
            elif isinstance(data_path, dict):
                print('Use data_path dict', flush=True)
                data_df = pd.concat([load_func(p, data_ratio*r) for p,r in data_path.items()], axis=0)
            elif isinstance(data_path, str):
                print('Use data_path str')
                data_df = load_func(data_path, data_ratio)
            else:
                print(f'Use data_path error, please check :{data_path}')
        else: print('Use data_df')
        
        # check exist and filesize
        print('='*10+' Before dataset_check, total old_len: ', data_df.shape[0], flush=True)
        def check_file_exist_and_filesize(row):
            img_path = row['img_path']
            if (not os.path.exists(img_path)) or (os.path.getsize(img_path)==0): 
                row['existed']=0  #;if phase=="Val" and "data_deepfake" in img_path:print("Non-existed img_path:", img_path, flush=True)
                print("Non-existed img_path:", img_path, flush=True)
            return row
        data_df['existed'] = 1
        data_df['img_path'] = data_df['img_path'].apply(lambda x: os.path.join(self.data_dir, x))
        data_df['target_binary'] = data_df['target'].apply(lambda x: 1 if x!=0 else 0)
        # from pandarallel import pandarallel  # 多进程处理df, faster
        # pandarallel.initialize(nb_workers=128)  # comment for save time
        # data_df = data_df.parallel_apply(lambda x: check_file_exist_and_filesize(x), axis=1)
        # data_df.drop(data_df[data_df['existed']==0].index, inplace=True)
        print('='*10+' After dataset_check, total new_len: ', data_df.shape[0], flush=True)
    
        if shuffle: data_df.sample(frac=1.0).reset_index(drop=True)
        self.img_paths = data_df['img_path'].tolist()
        self.labels = data_df['target'].tolist()
        self.labels_binary = data_df['target_binary'].tolist()

        siz = len(self.img_paths)
        cnt = Counter(self.labels_binary)
        print('='*20+f" Loaded {self.phase} total data:{siz}")
        print('='*20+f" Loaded {self.phase} 0_live:{cnt[0]}, 1_spoof:{cnt[1]}")
        print('='*20+f" Loaded {self.phase} multiclass distribution: \n{data_df['target'].value_counts()}")
        

    def __getitem__(self, index):
        if self.phase == 'Train':
            prefix_path = '/home/zoloz/8T-2/zhanyi/data/openset_df_competition/phase1/trainset/'
        elif self.phase == 'Val':
            prefix_path = '/home/zoloz/8T-2/zhanyi/data/openset_df_competition/phase1/valset/'
        else:
            prefix_path = '/home/zoloz/8T-2/zhanyi/data/openset_df_competition/phase1/valset/'
        img_path = os.path.join(prefix_path, self.img_paths[index])
        img_name = img_path.split('/')[-1]
        label = self.labels[index]
        # print('==> img_path:', img_path, flush=True)

        try:
            img1 = cv2.imread(img_path)
            assert img1 is not None

            # 兼容twoframe图像输入
            h,w = img1.shape[0],img1.shape[1]
            if (w//h)==2: 
                frame1 = img1[:,0:h,:]
                frame2 = img1[:,h:h*2,:]
                img1 = random.choice([frame1,frame2])
        
            img_inputs = {
                'img_frame1':np.ascontiguousarray(img1[:,:,::-1]),  # bgr2rgb
                'img_orig1': np.ascontiguousarray(img1[:,:,::-1]),
            }
            # img_inputs['img_fft1'] = torch.from_numpy(np.transpose(self.generate_FFT_3C(img1, (224,224)), (2, 0, 1))).float()
            # img_inputs['img_dct1'] = torch.from_numpy(np.transpose(self.generate_DCT_3C(img1, (224,224), color_type="YCrCb"), (2, 0, 1))).float()
            img_inputs['img_fft1'] = torch.from_numpy(np.zeros((3,224,224))).float()
            img_inputs['img_dct1'] = torch.from_numpy(np.transpose(self.generate_DCT_3C(img1, (224,224), color_type="YCrCb"), (2, 0, 1))).float()
            
            # img_depth = cv2.normalize(img_depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img_depth = torch.from_numpy(np.expand_dims(img_depth, axis=0)).float()  # 1HW
            # img_inputs['img_depth'] = img_depth
            img_inputs['img_depth'] = torch.from_numpy(np.zeros((3,224,224))).float()

            skip_norm_items = ['img_fft1', 'img_dct1', 'img_depth', 'img_mask']
            if self.transform is not None:
                for k,img_in in img_inputs.items():
                    if k in skip_norm_items: continue
                    img_in = Image.fromarray(img_in)
                    img_in = self.transform(img_in)
                    img_inputs[k] = img_in

            try:
                if self.with_mask and label!=0:  # read mask or not
                    # 超分后处理增强后的图像文件夹名称：
                    # fake__attedit_GPEN_FaceEnhancement__fake__swap_SimSwap__celebahq_crop_512__ant_blind_test_dataset_online/attedit_GPEN_FaceEnhancement__result_whole_swapsingle__01e7f9890f17158a9a1a38152b7b4bd4.jpg__03b4eed224662d5c3b26b8a38be744f8.jpg
                    # fake__attedit_Face-Super-Resolution__fake__swap_AIM__celebahq_crop_512__zhike_liveface_1000k_Apr18_part4_10w/attedit_Face-Super-Resolution__swap_aim_v1__00001874236edc4e4a074bbd94911b17.jpg__Jun11_00-38-33.jpg
                    mask_path = img_path
                    mask_path = mask_path.replace("/fake__attedit_GPEN_FaceEnhancement__", "/")
                    mask_path = mask_path.replace("/fake__attedit_Face-Super-Resolution__", "/")
                    mask_path = mask_path.replace("/attedit_GPEN_FaceEnhancement__", "/")
                    mask_path = mask_path.replace("/attedit_Face-Super-Resolution__", "/")
                    mask_path = mask_path.replace("/fake__", "/mask__")
                    mask = Image.open(mask_path).convert('L')
                    # 非1即0 的binary逻辑：窜改区域值为0，非窜改区域值为1
                else:
                    mask = Image.fromarray(np.ones((224,224))).convert('L')
            except Exception as e:
                print('Exception: ', e, flush=True)
                print('mask_path: ', mask_path, flush=True)
                mask = Image.fromarray(np.ones((224,224))).convert('L')
            mask = self.mask_transform(mask)  # apply image transformation
            img_inputs['img_mask'] = mask
            
            label_multiclass = label  # 通用Deepfake打标的是 多分类
            label_multiclass = torch.from_numpy(np.asarray(label_multiclass)).long()
            label = 1 if label!=0 else 0
            label = torch.from_numpy(np.asarray(label)).long()  # int64

            res_sample = {
                'img_inputs': img_inputs,
                'img_path':   img_path,
            }
            res_target = {
                'label':            label,
                'label_multiclass': label_multiclass,
            }
            result = {
                'samples_dict': res_sample,
                'targets_dict': res_target,
            }
            return result

        except Exception as e:
            print('Exception: ', e, flush=True)
            print('img_path: ', img_path, flush=True)
            return self.__getitem__((index + 1) % len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def load_dataset_csv(self, data_path='', data_ratio=1.0, sep=','):
        try: 
            df = pd.read_csv(data_path, sep=sep)[['img_path', 'target']]
        except:
            df = pd.read_csv(data_path, sep=sep, header=None, names={'img_path':str, 'target':int})
        print(f"loading from data_path: {data_path}", flush=True)
        print('Before df.dropna, df.shape: ', df.shape[0], flush=True)
        df.dropna(subset=['img_path', 'target'], inplace=True)
        print('After df.dropna, df.shape: ', df.shape[0], flush=True)
        df['img_path'] = df['img_path'].astype(str)
        df['target'] = df['target'].astype(int)
        if data_ratio>1:
            df = pd.concat([df]*int(data_ratio), axis=0).reset_index(drop=True)
        else:
            df = df.sample(frac=data_ratio).reset_index(drop=True)
        return df

    def gen_opticalflow(self, img1, img2):
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return self.draw_hsv(flow)

    def draw_hsv(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def np_normalize(self, a, print_flag=False):
        amin, amax = a.min(), a.max() # 求最大最小值
        if print_flag: print('Before normalize: ', amin, amax)
        b = (a-amin+1e-7) / (amax-amin+1e-7) # (矩阵元素-最小值)/(最大值-最小值)
        if print_flag: print('After normalize: ', b.min(), b.max())
        return b

    def generate_FFT_3C(self, image, size=None, shift=True):
        image = image.astype(np.float)
        img_ft = np.zeros(shape=size[::-1]+(3,), dtype=np.float)  # HW3
        for c in range(image.shape[-1]):
            img = image[..., c]
            f = np.fft.fft2(img, s=size[::-1])
            fshift = f  # no center shift
            if shift: fshift = np.fft.fftshift(f) 
            fimg = np.log(np.abs(fshift)+1)
            fimg = self.np_normalize(fimg)
            img_ft[..., c] = fimg
        return img_ft
    
    def generate_DCT_3C(self, image_bgr, size=None, color_type="YCrCb"):
        assert color_type in ["GRAY", "YUV", "RGB", "BGR", "YCrCb"]
        image = cv2.cvtColor(image_bgr, eval("cv2.COLOR_BGR2{}".format(color_type)))
        if size is not None: image = cv2.resize(image, size)
        image = image.astype(np.float)
        img_dct = np.zeros(shape=size[::-1]+(3,), dtype=np.float)
        img_dct[..., 0] = self.np_normalize(np.log(abs(cv2.dct(image[..., 0]))+1e-7))
        img_dct[..., 1] = self.np_normalize(np.log(abs(cv2.dct(image[..., 1]))+1e-7))
        img_dct[..., 2] = self.np_normalize(np.log(abs(cv2.dct(image[..., 2]))+1e-7))
        return img_dct

# 新建DataLoaderX类
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())
class DataLoaderX(DataLoader):
    def __iter__(self):
        return super().__iter__()


class PrefetchLoader():
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True

        for item_dict in iter(self.loader):
            with torch.cuda.stream(self.stream):
                #######################   custom by sankuai   ############################ 
                samples_dict = item_dict['samples_dict']
                targets_dict = item_dict['targets_dict']

                next_inputs = {}  # sample, sample_aux, sample_ft, sample_depth, sample_reflect...
                next_labels = {}  # target (Multi-label), target_bi (Binary-label)
                
                for k,v in samples_dict.items():
                    if isinstance(v, torch.Tensor): 
                        next_inputs[k] = v.cuda(non_blocking=True).float()
                    elif isinstance(v, dict): 
                        next_inputs[k] = {}
                        for kk,vv in v.items(): 
                            if isinstance(vv, torch.Tensor): 
                                next_inputs[k][kk]=vv.cuda(non_blocking=True).float()
                    else: next_inputs[k] = v
                
                for k,v in targets_dict.items():
                    if isinstance(v, torch.Tensor): 
                        next_labels[k] = v.cuda(non_blocking=True).long()
                    else: next_labels[k] = v
                #######################   custom by sankuai   ############################

            if not first:
                yield inputs, labels
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            inputs = next_inputs
            labels = next_labels

        yield inputs, labels

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class PrefetchLoader_Two():
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True

        for item_dict1,item_dict2 in zip(iter(self.loader1),iter(self.loader2)):
            with torch.cuda.stream(self.stream):
                #######################   custom by sankuai   ############################ 
                samples_dict1 = item_dict1['samples_dict']
                targets_dict1 = item_dict1['targets_dict']
                samples_dict2 = item_dict2['samples_dict']
                targets_dict2 = item_dict2['targets_dict']

                next_inputs = {}  # sample, sample_aux, sample_ft, sample_depth, sample_reflect...
                next_labels = {}  # target (Multi-label), target_bi (Binary-label)
                
                for k1,v1 in samples_dict1.items():
                    v2 = samples_dict2[k1]
                    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor): 
                        v1 = v1.cuda(non_blocking=True).float()
                        v2 = v2.cuda(non_blocking=True).float()
                        next_inputs[k1] = torch.cat((v1, v2))
                    elif isinstance(v1, dict) and isinstance(v2, dict): 
                        next_inputs[k1] = {}
                        for kk1,vv1 in v1.items(): 
                            vv2 = v2[kk1]
                            if isinstance(vv1, torch.Tensor) and isinstance(vv2, torch.Tensor): 
                                vv1 = vv1.cuda(non_blocking=True).float()
                                vv2 = vv2.cuda(non_blocking=True).float()
                                next_inputs[k1][kk1] = torch.cat((vv1, vv2))
                    else: next_inputs[k1] = [v1, v2]

                for k1,v1 in targets_dict1.items():
                    v2 = targets_dict2[k1]
                    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor): 
                        v1 = v1.cuda(non_blocking=True).long()
                        v2 = v2.cuda(non_blocking=True).long()
                        next_labels[k1] = torch.cat((v1, v2))
                    else: next_labels[k1] = [v1, v2]
                #######################   custom by sankuai   ############################

            if not first:
                yield inputs, labels
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            inputs = next_inputs
            labels = next_labels

        yield inputs, labels

    def __len__(self):
        return len(self.loader1)

    @property
    def sampler(self):
        return self.loader1.sampler

    @property
    def dataset(self):
        return self.loader1.dataset



def test_crop():
    data_path = './data/train_label.txt'
    fp = open(data_path, 'r')
    for line in fp:
        line = line.strip()
        img_path = line.split(' ')[0]
        label = int(line.split(' ')[1])

        img_org = cv2.imread(img_path)
        bbox_path = img_path.replace('.jpg', '_BB.txt').replace('.png', '_BB.txt')
        f = open(bbox_path)
        bbox = []
        # import ipdb; ipdb.set_trace()
        for line in f:
            line = line.strip()
            bbox = line.split(' ')
        f.close()
        real_h, real_w, real_c = img_org.shape
        x1 = (float(bbox[0])*(real_w / 224))
        y1 = (float(bbox[1])*(real_h / 224))
        w1 = (float(bbox[2])*(real_w / 224))
        h1 = (float(bbox[3])*(real_h / 224))
        score = float(bbox[4])
        x2 = x1 + w1
        y2 = y1 + h1

        ratio = 0.0625
        x1 = max(0, int(x1 - real_w*ratio))
        y1 = max(0, int(y1 - real_h*ratio))
        x2 = min(real_w, int(x2 + real_w*ratio))
        y2 = min(real_h, int(y2 + real_h*ratio))
        img_crop = img_org[y1:y2, x1:x2]
        import ipdb; ipdb.set_trace()

        # if label == 0:
        #     label = -1
        
        img_paths.append(img_path)
        labels.append(label)
    fp.close()

def tensor_to_img(img, mean=0, std=1):
    img = np.transpose(img.numpy(), (1, 2, 0))  # CHW->HWC
    # img = (img*std+ mean)*255
    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':

    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # dataset_check()
    # exit(0)    

    # test_crop()
    # exit(0)    


    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    transform = transforms.Compose([
                    # transforms.ToTensor(): Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: dummy_transform1(x)),
                    # transforms.Lambda(lambda x: dummy_transform2(x)),
                ])

    data_path = './data/train_label.txt'
    dataset = DatasetProcessing(data_path, transform=transform, phase='Train')
    # import ipdb; ipdb.set_trace()

    for i in range(len(dataset.img_filename)):
        img_path = dataset.img_filename[i]
        res = dataset.__getitem__(i)




    # # sampler = DummySampler(dataset)  #DummySampler(dataset)  #None #
    # loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)  ##shuffle=True #False
    #                      # pin_memory=True
    #                      # drop_last=False


    # # classes = dataset.classes
    # for epoch in range(10):
    #     print('epoch=%d -------------------------'%(epoch))
    #     for i, data in enumerate(loader, 0):
    #         print('i=%d: '%(i))

    #         # get the inputs
    #         images, labels = data
    #         # import ipdb; ipdb.set_trace()
    #         num = len(images)
    #         for n in range(num):
    #             image=images[n]
    #             label=labels[n]
    #             import ipdb; ipdb.set_trace()
    #             # print('\t\tlabel=%d : %s'%(label,classes[label]))
    #             #print('')
    #             img = tensor_to_img(image)
    #             # im_show('image', tensor_to_img(image), resize=6 )
    #             # cv2.waitKey(1)