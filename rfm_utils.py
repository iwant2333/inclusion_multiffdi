import torch
import torchvision
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2



def tensor2origimg(tensor, save_path):
    #save original img
    tensor = tensor.cpu().detach()
    mean = torch.as_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = torch.as_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    ori_img = tensor * std + mean  # in [0, 1]
    ori_img = ToPILImage()(ori_img[0, :])
    ori_img.save(save_path)
    return ori_img,save_path


def vis_grid_image(img_list, ncols=8, save_path='vis.jpg'):
    img_num = len(img_list)
    nrows = img_num//ncols
    nrows = nrows+1 if img_num%ncols!=0 else nrows

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np

    fig = plt.figure(33, figsize=(8., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1)
    for ax, im in zip(grid, img_list):
        ax.imshow(im)  # Iterating over the grid returns the Axes.

    plt.axis('off')
    plt.savefig(save_path)
    plt.close()



modelname = "xception"
resume_model = "./models/xbase_xception_model_batch_30000"

aug = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])])


def cal_normfam(model, inputs, cal_fam):
    fam = cal_fam(model, inputs)
    _, x, y = fam[0].shape
    fam = torch.nn.functional.interpolate(fam, (int(y/2), int(x/2)), mode='bilinear', align_corners=False)
    fam = torch.nn.functional.interpolate(fam, (y, x), mode='bilinear', align_corners=False)
    for i in range(len(fam)):
        fam[i] -= torch.min(fam[i])
        fam[i] /= torch.max(fam[i])
    return fam


def gen_heatmap(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    fam = heatmap + np.float32(image)
    return norm_image(fam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    image = image.copy()
    image -= np.min(image)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def parse_and_log_images(self, id_logs, x, y, y_hat, x2, title='', subscript=None, display_count=2):
    x = x[:,:3]  # ensure: N3HW
    y = y[:,:3]
    im_data = []
    for i in range(display_count):
        cur_im_data = {
            'input_face': common.log_input_image(x[i], self.opts),
            'target_face': common.tensor2im(y[i]),
            'output_face': common.tensor2im(y_hat[i]),
            'fakeb1_face': common.tensor2im(x2[i]),  # new added
        }
        if id_logs is not None:
            for key in id_logs[i]:
                # id_logs: 'diff_target', 'diff_input', 'diff_views', calc these based on ir_se50 ArcFace features
                cur_im_data[key] = id_logs[i][key]
        im_data.append(cur_im_data)
    self.log_images(title, im_data=im_data, subscript=subscript)
    self.log_images_seperately(title, im_data=im_data, subscript=subscript)



def log_images_seperately(self, name, im_data, subscript=None, log_latest=False):
    step = self.global_step if not log_latest else 0
    subscript = subscript if subscript is not None else ""

    display_count = len(im_data)
    for i in range(display_count):
        hooks_dict = im_data[i]
        
        save_dir  = name.replace('/test/', '/test_seperately/01_Input/')
        save_dir  = os.path.join(self.logger.log_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_name = '{}_{:04d}.jpg'.format(subscript, step)
        save_path = os.path.join(save_dir, save_name)
        hooks_dict['input_face'].save(save_path)

        save_dir  = name.replace('/test/', '/test_seperately/02_Target/')
        save_dir  = os.path.join(self.logger.log_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_name = '{}_{:04d}.jpg'.format(subscript, step)
        save_path = os.path.join(save_dir, save_name)
        hooks_dict['target_face'].save(save_path)
        
        save_dir  = name.replace('/test/', '/test_seperately/03_Output/')
        save_dir  = os.path.join(self.logger.log_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_name = '{}_{:04d}.jpg'.format(subscript, step)
        save_path = os.path.join(save_dir, save_name)
        hooks_dict['output_face'].save(save_path)

        save_dir  = name.replace('/test/', '/test_seperately/04_fakeB1/')
        save_dir  = os.path.join(self.logger.log_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_name = '{}_{:04d}.jpg'.format(subscript, step)
        save_path = os.path.join(save_dir, save_name)
        hooks_dict['fakeb1_face'].save(save_path)


        hooks_simi = '{} {:.4f}\n'.format(save_name, float(hooks_dict['diff_target']))
        save_dir  = name.replace('/test/', '/test_seperately/')
        save_dir  = os.path.join(self.logger.log_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_name = 'similarity.txt'
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, 'a+') as f:
            f.write(hooks_simi)






