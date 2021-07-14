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

Run `req.sh` in the activated conda environment to setup requirements.

You can follow a procedure similar to [this guide](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md) to install OpenMMLab libraries. 

[Download pretrained models](https://drive.google.com/drive/folders/11u3pvnoTxQb_u8i989zFG5ejIu7-l3Ax?usp=sharing) for the logo classifier and move them into `pretrains` folder.

**! Currently, only .ipynb files are working properly, parser for .py files will be added later on.**

For logo classification, perform the steps in `Augmentations & Split.ipynb` and `LogoClassifier.ipynb` files consequently. Organize a `logo_data` directory where the subdirectories named according to companies. These subdirectories should include logo images of specified companies.

For OCR tool, perform the steps in `OCR Tool.ipynb` file.
