# LogoClassifier

Logo classifier written utilizing OpenMMLab MMOCR, MMClassification tools and PyTorch.

For custom datasets, [OpenMMLab new dataset guide](https://github.com/open-mmlab/mmclassification/blob/master/docs/tutorials/new_dataset.md)  is followed. The custom dataset file provided is `datasets/logolist.py` and it should be moved to `mmcls/datasets` folder once the required openmmlab libraries are installed.

Before installing the requirements, setup anaconda and create a new conda environment via 

`conda create -n logo_cls python=3.7 -y
 conda activate logo_cls
 conda install pytorch torchvision -c pytorch`
 
You can follow a procedure similar to [this guide](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md). 

Run `req.sh` in the activated conda environment to setup requirements.
