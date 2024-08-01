import argparse, glob, os, cv2, sys, random
from asyncio.unix_events import DefaultEventLoopPolicy
from matplotlib.pyplot import title
import pandas as pd
import hashlib, copy
import matplotlib
matplotlib.use('agg')


def iter_folder_files(folder='', suffix=None):
    '''iterate folder files with xxx suffix'''
    file_paths = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        # print(rel_path)
        
        for f in files:
            if (suffix is not None) and (f.split('.')[-1] not in suffix): continue
            file_paths.append(os.path.join(root, f))
    
    # return sorted(file_paths)[:100]  # just for debug
    return sorted(file_paths)


def vis_grid_image(img_list, ncols=8, save_path='vis.jpg', fig_title='', img_size=512):
    # vis img_list=List[cv2 images rgb]
    img_num = len(img_list)
    if img_num <= 0: return

    for i,img in enumerate(img_list):
        img_list[i] = cv2.resize(img, (img_size, img_size))

    ncols = min(ncols, len(img_list))
    nrows = img_num//ncols
    nrows = nrows+1 if img_num%ncols!=0 else nrows

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(33, figsize=(ncols*1.5, nrows*1.5))
    fig.suptitle(fig_title, fontsize=16)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1)
    for ax, im in zip(grid, img_list):
        ax.imshow(im, interpolation='nearest', aspect='auto')  # Iterating over the grid returns the Axes.
        ax.set_xticks([])  # close axis ticks
        ax.set_yticks([])

    plt.axis('off')
    plt.savefig(save_path, dpi=300)  # 建议将dpi设置在150到300之间 更清晰
    plt.close()


def vis_data_dir_example(data_dir="", data_num=16):
    # 将某个 data_dir文件夹中 图像拼接成一张大图 便于显示

    tiny_dir = data_dir
    print(f"create vis big map for: {tiny_dir}")
    path_list = iter_folder_files(tiny_dir, suffix=['jpg','png','jpeg','JPG'])
    path_list = path_list[:min(data_num, len(path_list))]

    save_dir = os.path.join(data_dir, "aa_tinyvis_bigmap")
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{tiny_dir.split('/')[-1]}.jpg"
    img_list = [cv2.imread(p)[:,:,::-1] for p in path_list]
    # print(f"save_path: {save_path}");continue
    
    vis_grid_image(img_list, ncols=8, save_path=save_path, fig_title=tiny_dir.split('/')[-1])
    print(f"saved to: {save_path}")

    return 
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create imglist file')
    parser.add_argument('--data_dir', required=True, help='a common image directory to create vis big map')
    parser.add_argument('--data_num', type=int, default=16, help='image nums to create vis big map')
    args = parser.parse_args()

    vis_data_dir_example(args.data_dir, args.data_num)
    

