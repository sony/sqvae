# SQ-VAE (Vision Dataset)

This repository contains the official Pytorch implementation of SQ-VAE.

## Training
The training of a model can be done by calling main.py with the corresponding yaml file. The list of yaml files can be found below.
Please refer to main.py (or execute 'python main.py --help') for the usage of extra arguments.

### Setup steps before training of a model
* Set the checkpoint path "_C.path" (/configs/defaults.py:4) 
* Set the dataset path, "_c.path_dataset" (/configs/defaults.py:5).

### Train a model
Example 1: Gaussian SQ-VAE (I) on CIFAR10
```
python main.py -c "cifar10_gauss_1.yaml" --save
```
Example 2: Gaussian SQ-VAE (III) on CelebA<sup>1</sup>
```
python main.py -c "celeba_gauss_3.yaml" --save
```
Example 3: vMF SQ-VAE on CelebAMask<sup>2</sup>
```
python main.py -c "celebamask_vmf.yaml" --save
```


### Where to find the checkpoints
If the trainning is successful, checkpoint folders will be generated under the folder (cfgs represents the yaml file specified when calling main.py):
```
configs.defaults._C.path + '/' + cfgs.path_spcific
```


### List of yaml files: models work on continuous/discrete data distributions
| Config file | Description |
|---|---|
| mnist_gauss_1.yaml | Gaussian SQ-VAE (I) on MNIST |
| fashion-mnist_gauss_1.yaml | Gaussian SQ-VAE (I) on Fashion-MNIST |
| cifar10_gauss_1.yaml | Gaussian SQ-VAE (I) on CIFAR10 |
| celeba_gauss_1.yaml | Gaussian SQ-VAE (I) on CelebA |
| celeba_gauss_2.yaml | Gaussian SQ-VAE (II) on CelebA |
| celeba_gauss_3.yaml | Gaussian SQ-VAE (III) on CelebA |
| celeba_gauss_4.yaml | Gaussian SQ-VAE (IV) on CelebA |
| celebamask_vmf.yaml | vMF SQVAE on CelebAMask |

The major difference among SQ-VAE (I)-(IV) is the form of the covariance matrix. Please also refer to Table 1 in our paper for details.

<sup>1</sup>*We recommend you download CelebA dataset directly from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) instead of using torchvision.*

<sup>2</sup>*The dataset for this task is the result of the face parsing on CelebAMask. The face parsing script can be found in the Acknowledgement section.*


## Experiments
"[checkpoint_foldername_with_timestep]" means the folder names under the path "[configs.defaults._C.path + '/' + cfgs.path_spcific]".
These folder names are consist of the model names, the seed indices and the timestamps.

## Dependencies
numpy
scipy
torch
torchvision
PIL

## Acknowledgements
The scripts for data processing of CelebAMask-HQ are available in
https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing

The scripts for calculation of the modified Bessel functions are adopted from
https://github.com/nicola-decao/s-vae-pytorch

The scripts for mIoU calculation are adopted from
https://github.com/Tramac/awesome-semantic-segmentation-pytorch



## Citation
```
@INPROCEEDINGS{takida2022sq-vae,
    author={Takida, Yuhta and Shibuya, Takashi and Liao, WeiHsiang and Lai, Chieh-Hsin and Ohmura, Junki and Uesaka, Toshimitsu and Murata, Naoki and Takahashi Shusuke and Kumakura, Toshiyuki and Mitsufuji, Yuki},
    title={SQ-VAE: Variational Bayes on Discrete Representation with Self-annealed Stochastic Quantization},
    booktitle={International Conference on Machine Learning},
    year={2022},
    }
```