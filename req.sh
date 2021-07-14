conda install -c conda-forge opencv
conda install -c conda-forge albumentations
conda install -c conda-forge pytest-shutil
conda install -c anaconda pillow
conda install -c conda-forge tqdm
conda install -c conda-forge glob2
conda install -c conda-forge cairosvg
conda install -c jmcmurray json
conda install -c anaconda urllib3
conda install -c anaconda csvkit
conda install -c hellock icrawler
conda install -c anaconda requests
conda install -c auto html

pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmdet
pip install mmocr
pip install mmcls


mkdir temp
cd temp
git clone https://github.com/open-mmlab/mmocr.git .
cd ..
mv temp/configs configs/configs_ocr
rm -rf temp

mkdir temp
cd temp 
git clone https://github.com/open-mmlab/mmclassification.git .
cd ..
mv temp/configs configs/configs_cls
mv temp/tools configs/cls_tools
rm -rf temp

mkdir data/logo_data

mkdir data/logo_data/train
mkdir data/logo_data/val
mkdir data/logo_data/meta

mkdir data/predictions
mkdir data/form_image_jsons
