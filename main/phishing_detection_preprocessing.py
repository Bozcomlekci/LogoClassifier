import csv
import requests
import os
import urllib
from html.parser import HTMLParser
from heapq import merge
from icrawler.builtin import BingImageCrawler,BaiduImageCrawler
import json
import time as time
import signal


ids = list(csv.reader(open('datasets/phishingFormIDs.csv'), delimiter=','))
names = list(csv.reader(open('datasets/phishingCompanyNames.csv'), delimiter=','))


ids = [id[0] for id in ids][1:-1]
names = [name[0] for name in names][1:-1]


print(ids)
print(names)

with open("data/api_key.json", "r") as fp:
    api_json = json.load(fp)

api_key = api_json['api_key']


formDict = {}
#cnt = 0
for id in ids:
    address = "https://api.jotform.com/form/" + id +  "/properties?apiKey={}".format(api_key)
    print(address)
    res = requests.get(address)
    formDict[id] = res.json()
    #cnt = cnt + 1
    #if cnt == 100:
    #    break


for name in names:
    newpath = os.path.join(os.getcwd(), 'logos' )
    newpath = os.path.join(newpath, name )
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = os.path.join(newpath, 'vectorized')    
    if not os.path.exists(newpath):
        os.makedirs(newpath)


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'img': 
            yield( attrs )

    def handle_endtag(self, tag):
        if tag == 'img':
            None

    def handle_data(self, data):
        None
parser = MyHTMLParser()


formImages = {}
pred = {}
allFormatList = ['.png', '.jpg', '.jpeg', '.svg', '.apng']
for key, exp in formDict.items():
    exp_2 = exp['content']
    if isinstance(exp_2, list):
        pred[key] = 0
    elif isinstance(exp_2, dict) and ('emails' not in exp_2.keys()):
        pred[key] = 0
    else:
        pred[key] = 0
        refined = exp_2['emails']
        imageList = []
        formatList = []
        imageInfo = []
        for litem in refined:
            if isinstance(litem, str):
                formContent = litem
            else:    
                formContent = litem['body']
            if formContent != None:
                start = [i.start() for i in re.finditer('https', formContent)]
                for s in start:
                    e = formContent.find('"', s)
                    ext = formContent[e-4:e]
                    ext2 = formContent[e-5:e]
                    link = formContent[s:e]
                    if (ext in allFormatList) and (link != 'https://cdn.jotfor.ms/assets/img/builder/email_logo_small.png'):
                        pred[key] = 1
                        if link not in imageList: 
                            imageList.append(link)
                            formatList.append(ext)
                    elif (ext2 in allFormatList) and (link != 'https://cdn.jotfor.ms/assets/img/builder/email_logo_small.png'):
                        pred[key] = 1                       
                        if link not in imageList:    
                            imageList.append(link)
                            formatList.append(ext2)
        if pred[key] == 1:
            imageInfo = []
            imageInfo.append(imageList)
            imageInfo.append(formatList)
            formImages[key] = imageInfo
            print(imageInfo)

#with open("formImageLinks.json", "w") as fp:
    #json.dump(formImages, fp)


#Load from saved form links
with open("formImageLinks.json", "r") as fp:
    formImages = json.load(fp)


#with open('pred.json', 'w') as fp:
    #json.dump(pred, fp)


with open("pred.json", "r") as fp:
    pred = json.load(fp)


"""for name in names: 
    keyword = name + ' logo'
    logo_folder = 'logo_data/' + name
    crawler = BingImageCrawler(storage={'root_dir': logo_folder})
    crawler.crawl(keyword=keyword, max_num=100)
    print('Crawled images for {} downloaded'.format(name) )"""

image_formats = ("image/png", "image/jpeg", "image/jpg", "image/apng", "image/svg")

def excepthandler(signum, frame):
    raise Exception('Cannot retrieve the image in given time')

formImg = {}
root = os.path.join(os.getcwd(), 'form_images' )
exceptionForms = {}
for form_id, imageinfos in formImages.items():
    j = 0 
    newpath = os.path.join(root, form_id)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    imgDataList = []
    images, formats = imageinfos
    exceptionLinks = []
    for image_link, image_format in zip(images, formats):
        print(image_link)
        signal.signal(signal.SIGALRM, excepthandler)
        signal.alarm(10)
        image_name = 'form_images/' + str(form_id) + '/' + form_id + '_' + str(j) 
        try:
            image_name_png = image_name + image_format
            img_data = requests.get(image_link).content
            with open(image_name_png, 'wb') as handler:
                handler.write(img_data) 
                imgDataList.append(img_data)
            j += 1
        except:
            exceptionLinks.append(image_link)
    formImg[form_id] = imgDataList
    if exceptionLinks != []:
        exceptionForms[form_id] = exceptionLinks
    print('{} is processed'.format(form_id))
    
with open("exceptionLinks.json", "w") as fp:
    json.dump(exceptionLinks, fp)
    
signal.alarm(0)
print('Processing images in forms finished')


excpt_json = "others/exceptionLinks.json"
if not os.path.exists(excpt_json):
    with open(excpt_json, "w") as fp:
        json.dump(exceptionForms, fp)

#Exception throwing images in links related to forms are downloaded by hand
with open(excpt_json, "r") as fp:
    exceptionForms = json.load(fp)

print(exceptionForms)

