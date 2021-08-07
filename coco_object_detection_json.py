import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
import json
from pycocotools import mask
from xml.dom import minidom
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# Convert VOC annotations files into COCO format


def generateVOC2Json(rootDir, xmlFiles):
    attrDict = dict()
    attrDict["categories"] = [{"supercategory": "none", "id": 0, "name": "behind"}]
    images = list()
    annotations = list()
    for count,file in enumerate(xmlFiles):
        image_id = count + 1
        image = dict()

        Tree = ET.parse(rootDir + file)
        root=Tree.getroot()
        image['file_name'] = str(root.find('filename').text)
        image['height'] = int(root.find('imagesize').find('nrows').text)
        image['width'] = int(root.find('imagesize').find('ncols').text)
        image['sem_seg_file_name'] = 'trimaps/' + file[:-4] + '.jpg'
        image['id'] = image_id

        print("File Name: {} and image_id {}".format(file, image_id))
        images.append(image)

        id1 = 1
        for object in root.findall('object'):
            for value in attrDict["categories"]:
                annotation = dict()
                if str(object.find('name').text) == value["name"]:
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = image_id
                    points=[]
                    for point in object.find('polygon').findall('pt'):
                        x=float(point.find('x').text)
                        y=float(point.find('y').text)
                        points.append((x,y))
                    points=list(set(points))
                    points=sorted(points)
                    x1 = int(points[0][0])
                    y1 = int(points[0][1])
                    x2 = int(points[2][1])
                    y2 = int(points[1][1])
                    if x1>x2:
                        temp=x1
                        x1=x2
                        x2=temp
                    if y1>y2:
                        temp=y1
                        y1=y2
                        y2=temp

                    witdth=x2-x1
                    height=y2-y1
                    annotation["bbox"] = [x1, y1,witdth,height]
                    annotation["area"] = float(witdth * height)
                    annotation["category_id"] = value["id"]
                    annotation["ignore"] = 0
                    annotation["id"] = image_id

                    annotation["segmentation"] = []
                    id1 += 1

                    annotations.append(annotation)


        attrDict["images"] = images
        attrDict["annotations"] = annotations
        attrDict['info'] = {
            'contributor': 'QuangPham',
            'date_created': '2020/05/05',
            'description': 'Pets',
            'url': 'https://viblo.asia/u/QuangPH',
            'version': '1.1',
            'year': 2020
        }

    attrDict['licenses'] = [{'id': 1, 'name': 'QuangPham', 'url': 'https://viblo.asia/u/QuangPH'}]
    jsonString = json.dumps(attrDict)

    return jsonString



# split train/test
# trainFile = "./annotations/trainval.txt"
XMLFiles = list()
for index,file in enumerate(os.listdir("behind_xml_img/xml/")):
    XMLFiles.append(file)


trainXMLFiles, testXMLFiles = train_test_split(XMLFiles, test_size=0.2, random_state=24)
print(len(trainXMLFiles), len(testXMLFiles))

rootDir = "behind_xml_img/xml/"
train_attrDict = generateVOC2Json(rootDir, trainXMLFiles)
with open("coco_detection_json/train_object_detection.json", "w") as f:
    f.write(train_attrDict)

test_attrDict = generateVOC2Json(rootDir, testXMLFiles)
with open("coco_detection_json/test_object_detection.json", "w") as f:
    f.write(test_attrDict)