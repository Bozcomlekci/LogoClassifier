# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(mmcv.__version__)
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check mmocr installation
import mmocr
print(mmocr.__version__)

from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.pipelines.crop import crop_img

import json
import os

import cairosvg 


# ## OCR Tool with MMOCR

############################OPTIONALLY ADJUST CONFIG FILES###############################
#An extensive list of configurations can be found in https://github.com/open-mmlab/mmocr/tree/main/configs
det_model = 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py'
recog_model = 'robust_scanner/robustscanner_r31_academic.py'

#Pretrained OCR models for detection and recognition
det_model_pth = 'drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
recog_model_pth = 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
#########################################################################################

det_download_root = 'https://download.openmmlab.com/mmocr/textdet' 
recog_download_root = 'https://download.openmmlab.com/mmocr/textrecog' 

det_config = os.path.join('configs/configs_ocr/textdet', det_model)
recog_config = os.path.join('configs/configs_ocr/textrecog', recog_model)

det_ckpt = os.path.join(det_download_root, det_model_pth)
recog_ckpt = os.path.join(recog_download_root, recog_model_pth)


print(det_config)
print(recog_config)
print(det_ckpt)
print(recog_ckpt)


def det_and_recog_inference(img, det_model, recog_model):
    image_path = img
    end2end_res = {'filename': image_path}
    end2end_res['result'] = []

    image = mmcv.imread(image_path)
    det_result = model_inference(det_model, image)
    bboxes = det_result['boundary_result']

    box_imgs = []
    for bbox in bboxes:
        box_res = {}
        box_res['box'] = [round(x) for x in bbox[:-1]]
        box_res['box_score'] = float(bbox[-1])
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        box_img = crop_img(image, box)
        recog_result = model_inference(recog_model, box_img)
        text = recog_result['text']
        text_score = recog_result['score']
        if isinstance(text_score, list):
            text_score = sum(text_score) / max(1, len(text))
        box_res['text'] = text
        box_res['text_score'] = text_score

        end2end_res['result'].append(box_res)

    return end2end_res

#!python demo/ocr_image_demo.py /content/Invoice.png demo/output.jpg

def detectLogoText( img, out_file, 
                 det_config ='./configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py',
                 det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth',
                 recog_config = './configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
                 recog_ckpt = 'https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth',
                 device = 'cuda:0'
                ):
    # build detect model
    detect_model = init_detector(det_config, det_ckpt, device=device)
    if hasattr(detect_model, 'module'):
        detect_model = detect_model.module
    if detect_model.cfg.data.test['type'] == 'ConcatDataset':
        detect_model.cfg.data.test.pipeline =             detect_model.cfg.data.test['datasets'][0].pipeline

    # build recog model
    recog_model = init_detector(recog_config, recog_ckpt, device=device)
    if hasattr(recog_model, 'module'):
        recog_model = recog_model.module
    if recog_model.cfg.data.test['type'] == 'ConcatDataset':
        recog_model.cfg.data.test.pipeline =             recog_model.cfg.data.test['datasets'][0].pipeline

    det_recog_result = det_and_recog_inference(img, detect_model, recog_model)
    print(f'result: {det_recog_result}')
    mmcv.dump(
        det_recog_result,
        out_file + '.json',
        ensure_ascii=False,
        indent=4)

    img = det_recog_show_result(img, det_recog_result)
    mmcv.imwrite(img, out_file)


# In[ ]:


def perform_ocr():
    with open("data/predictions/pred.json", "r") as fp:
        pred = json.load(fp)

    root = 'form_images'
    out_root = 'ocr_results'
    for form_id, val in pred.items():
        if val == 1:
            print('Processing {}'.format(form_id))
            in_path = os.path.join(root, form_id)
            out_path = os.path.join(out_root, form_id)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            img_num = 1
            for img_name in os.listdir(in_path):
                img = os.path.join(in_path, img_name)
                out_dir = os.path.join(out_root, str(form_id), str(img_num))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out = os.path.join(out_dir, img_name)
                try:
                    detectLogoText( img, out, det_config=det_config, recog_config=recog_config, 
                                det_ckpt=det_ckpt, recog_ckpt=recog_ckpt)
                except:
                    pred[form_id] = -1
                    #img_new = img[:-4]+'.png'
                    #cairosvg.svg2png(url=img, write_to=img_new)
                    #detectLogoText( img_new, out, det_config=det_config, recog_config=recog_config, 
                    #            det_ckpt=det_ckpt, recog_ckpt=recog_ckpt)
                img_num += 1
    print('OCR results finished, recording predictions back...')            
    with open("data/predictions/pred_refined.json", "w") as fp:
        json.dump(pred,fp)



perform_ocr()


# #### Prediction of forms with N/A files (svg, broken images, images that can't be read) printed


'''with open("pred_refined.json", "r") as fp:
    pred_refined = json.load(fp)

for form_id, val in pred_refined.items():
    if val == -1:
        print(form_id)'''


# ### Record similar words to compare with OCR results

similar_words = { 
                 'adobe': ['adobe'],
                 'airbnb': ['airbnb','air', 'bnb'],
                 'alibaba': ['alibaba','ali', 'baba'] ,
                 'amazon': ['amazon','amazon services'],
                 'american express': ['american express','american', 'express'],
                 'apple': ['apple'],
                 'at&t': ['at&t'],
                 'bank of america': ['bank of america'],
                 'chase bank': ['chase bank', 'chase'],
                 'dhl': ['dhl'],
                 'ebay': ['ebay'],
                 'facebook': ['facebook'],
                 'fedex': ['fedex', 'federal express'],
                 'google': ['google'],
                 'hsbc': ['hsbc'],
                 'ibm':['ibm'],
                 'icbc':['icbc'],
                 'ikea':['ikea'],
                 'jpmorgan': ['jpmorgan', 'jp', 'morgan'],
                 'mastercard': ['mastercard', 'master', 'card'],
                 'microsoft': ['microsoft', 'micro', 'soft'],
                 'netflix': ['netflix'],
                 'oracle': ['oracle'],
                 'pwc': ['pwc'],
                 'samsung': ['samsung'],
                 'spotify': ['spotify'],
                 'square': ['square'],
                 'stripe': ['stripe'],
                 'ups': ['ups'],
                 'usps': ['usps', 'u.s.', 'us mail', 'united states postal service'],
                 'visa': ['visa']   
               }

with open("data/similar_words.json", "w") as fp:
    json.dump(similar_words, fp)

