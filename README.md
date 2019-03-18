# Vggvox-TensorFlow

This model is for speaker identification.

This model could be trained and test pre-trained on the [VoxCeleb(1)](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets as described in the following paper:

```
[1] A. Nagrani*, J. S. Chung*, A. Zisserman, VoxCeleb: a large-scale speaker identification dataset, 
INTERSPEECH, 2017
```



### Dependencies

[1] tensorflow-gpu=1.11.0 
[1] librosa=0.6.3 
[2] scipy=1.1.0   

### Train

Before training and testing, you must prepare the dataset and configure the environment well.

The full dataset can be freely downloaded from [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

After get the full dataset, and you can see \[utils/vox1_split_backup.txt\] in this repo so just prepare a gpu environment, get start training!

Training:

> $ python3 train.py train --voxceleb_wav_dir '/data/ChuyuanXiong/up/wav/' --vox_split_txt_file 'utils/vox1_split_backup.txt' --batch_size 32 --lr 0.001 --ckpt_save_dir '/data/ChuyuanXiong/backup/speaker_real318_ckpt' 

The args:

1. --voxceleb_wav_dir: After you get full dataset, you will find all data is in a wav dir. Record the dir path.
2. --vox_split_txt_file: You can find this file in \[utils/vox1_split_backup.txt\], and you can also find it in [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
3. --batch_size: Batch size.
4. --lr: Learning rate. The default optimizer is Adam.
5. --ckpt_save_dir: Where you want to save the ckpt files. Default max ckpt files is 3.

### Test

Before test, you must have the pre-trained ckpt file. 

Test:

> $ python3 train.py test --voxceleb_wav_dir '/data/ChuyuanXiong/up/wav/' --vox_split_txt_file 'utils/vox1_split_backup.txt' --batch_size 32 --ckpt_restore_file '/data/ChuyuanXiong/backup/triplet_backup2/Speaker_vox_iter_51500.ckpt' --random_seed 100

The args:

1. --voxceleb_wav_dir: After you get full dataset, you will find all data is in a wav dir. Record the dir path.
2. --vox_split_txt_file: You can find this file in \[utils/vox1_split_backup.txt\], and you can also find it in [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
3. --batch_size: Batch size.
5. --ckpt_restore_file: Pre-trained model.


