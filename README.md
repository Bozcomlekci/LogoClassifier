# LogoClassifier

Logo classifier written utilizing OpenMMLab MMOCR, MMClassification tools and PyTorch.

For custom datasets, [OpenMMLab new dataset guide](https://github.com/open-mmlab/mmclassification/blob/master/docs/tutorials/new_dataset.md)  is followed. The custom dataset file provided is `datasets/logolist.py` and it should be moved to `mmcls/datasets` folder once the required openmmlab libraries are installed.

Before installing the requirements, setup anaconda and create a new conda environment via 

```shell
conda create -n logo_cls python=3.7 -y
conda activate logo_cls 
```
 
and install PyTorch via
 
```shell
conda install pytorch=1.8.0 torchvision=0.2.1 torchaudio=0.8.0 cudatoolkit=11.1 -c pytorch
```
 
You can follow a procedure similar to [this guide](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md). 

Run `req.sh` in the activated conda environment to setup requirements.
