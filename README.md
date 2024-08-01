
## Usage

### Traing

1.Download the training data and put the decompressed data into the data folder, then Change the data path in train.py (row 263) and dataset_processing.py (row 121).
  ```
  data
    multiFFDI
      phase1
        - trainset
        - valset
        trainset_label.txt
        valset_label.txt
  ```
  train.py (row 263)
  ```
  data_root = "./data/multiFFDI/phase1/" # your data path
  ```
  dataset_processing.py (row 121)
  ```
  prefix_path = "./data/multiFFDI/phase1/trainset/" # your data path
  ```
2.Change the gpu id in train_main.py (row 67).

3.Run the training script:

  ```
  sh run_train.sh
  ```

  After training, the model parameter weights are saved in 'classifier' folder.


### Pretrained Models
Please download our pre-trained model [multiffdi](https://drive.google.com/file/d/1JBB7y__ii1FjdzXaHE3d1bekgp2dJm0h/view?usp=drive_link) and put it in ```./classifier/for_full_exp_general/inclusion_coatnet/Jul02_09-33-49/```.


### Inference

  Finally, test the model, you can change you own model weights path in test.py (row 58), then you can choose an image and update the image path in test.py (row 64), and run ```python test.py``` to obtain the image score.

### Image Examples
<!-- ![image](./imgs/88b8a37897d79f59890432dff0107633.jpg =200x200)
![image](./imgs/7ab02a23cf8a30cda4c70647b6c34e4b.jpg =200x200)
![image](./imgs/e1a9bb09f6c3d79374c658c145a8d708.jpg =200x200) -->
<img src='./imgs/88b8a37897d79f59890432dff0107633.jpg' width='224' height='224'>
<img src='./imgs/7ab02a23cf8a30cda4c70647b6c34e4b.jpg' width='224' height='224'>
<img src='./imgs/e1a9bb09f6c3d79374c658c145a8d708.jpg' width='224' height='224'>

## Acknowledgements
This codebase was bulit upon [CoAtNet](https://github.com/chinhsuanwu/coatnet-pytorch). We thank the authors for making those repositories public.