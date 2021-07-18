# import dota_utils as util
import os
# import cv2
import json
# from PIL import Image
import xmltodict
import xml.etree.ElementTree as ET
# from ShipRSImageNet_devkit import ShipRSImageNet_utils as util
# from collections import OrderedDict

wordname_50 = ['Other Ship', 'Other Warship', 'Submarine', 'Other Aircraft Carrier', 'Enterprise', 'Nimitz', 'Midway',
           'Ticonderoga',
           'Other Destroyer', 'Atago DD', 'Arleigh Burke DD', 'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Other Frigate',
           'Perry FF',
           'Patrol', 'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 'Osumi LL',
           'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Commander', 'Other Auxiliary Ship', 'Medical Ship', 'Test Ship',
           'Training Ship',
           'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF', 'Other Merchant', 'Container Ship', 'RoRo', 'Cargo',
           'Barge', 'Tugboat', 'Ferry', 'Yacht', 'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft',
           'Motorboat', 'Dock']
# wordname_50 = ['Other Ship', 'Other Warship', 'Submarine', 'Other Aircraft Carrier', 'Enterprise', 'Nimitz', 'Midway',
#            'Ticonderoga',
#            'Other Destroyer', 'Atago DD', 'Arleigh Burke DD', 'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Frigate',
#            'Perry FF',
#            'Patrol', 'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 'Osumi LL',
#            'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Commander', 'Other Auxiliary Ship', 'Medical Ship', 'Test Ship',
#            'Training Ship',
#            'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF', 'Other Merchant', 'Container Ship', 'RoRo', 'Cargo',
#            'Barge', 'Tugboat', 'Ferry', 'Yacht', 'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft',
#            'Motorboat', 'Dock']

def ShipImageNet2COCOTrain(filenames, destfile, cls_names, level_num):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    # imageparent = os.path.join(srcpath, 'JPEGImages')
    # labelparent = .path.join(srcpath, 'Annotations_v2')

    if level_num == 3:
        level_class = 'level_3'
    elif level_num == 2:
        level_class = 'level_2'
    elif level_num == 1:
        level_class = 'level_1'
    else:
        level_class = 'level_0'

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            doc = xmltodict.parse(open(file).read())
            tree = ET.parse(file)
            root = tree.getroot()

            single_image = {}
            single_image['file_name'] = str(doc['annotation']['filename'])
            single_image['id'] = image_id
            single_image['width'] = int(doc['annotation']['size']['width'])
            single_image['height'] = int(doc['annotation']['size']['height'])
            # print(single_image)
            data_dict['images'].append(single_image)

            # annotations
            for obj in root.iter('object'):
                single_obj = {}
                single_obj['area'] = float(obj.find('Ship_area').text)
                single_obj['category_id'] = int(obj.find(level_class).text)
                single_obj['segmentation'] = []
                x1 = float(obj.find('polygon').find("x1").text)
                y1 = float(obj.find('polygon').find("y1").text)
                x2 = float(obj.find('polygon').find("x2").text)
                y2 = float(obj.find('polygon').find("y2").text)
                x3 = float(obj.find('polygon').find("x3").text)
                y3 = float(obj.find('polygon').find("y3").text)
                x4 = float(obj.find('polygon').find("x4").text)
                y4 = float(obj.find('polygon').find("y4").text)
                single_obj['segmentation'] = x1, y1, x2, y2, x3, y3, x4, y4
                single_obj['iscrowd'] = 0
                xmin = int(obj.find('bndbox').find("xmin").text)
                ymin = int(obj.find('bndbox').find("ymin").text)
                xmax = int(obj.find('bndbox').find("xmax").text)
                ymax = int(obj.find('bndbox').find("ymax").text)

                width, height = xmax - xmin, ymax - ymin
                # 计算旋转矩形框旋转角度
                # roted_box = util.polygonToRotRectangle([x1,y1,x2,y2,x3,y3,x4,y4])
                # xcenter,ycenter,width,height,angle = roted_box
                single_obj['bbox'] = xmin,ymin,width,height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)
    print('Total Instances:',image_id)

def ShipImageNet2COCOTest(filenames, destfile, cls_names):
    # imageparent = os.path.join(srcpath, 'JPEGImages')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            doc = xmltodict.parse(open(file).read())
            single_image = {}
            single_image['file_name'] = str(doc['annotation']['filename'])
            single_image['id'] = image_id
            single_image['width'] = int(doc['annotation']['size']['width'])
            single_image['height'] = int(doc['annotation']['size']['height'])
            data_dict['images'].append(single_image)
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def get_filenames(rootdir, file_dir, set_name):
    dataset_name = set_name + '.txt'
    File = os.path.join(text_dir, dataset_name)
    filenames = list()
    level_num = 3
    with open(File, "rb") as f:
        for line in f:
            fileName = str(line.strip(), encoding="utf-8")
            # print(fileName)
            fle_xml = fileName.replace('.bmp', '.xml')
            annotation_path = os.path.join(rootdir, fle_xml)
            filenames.append(annotation_path)
    return filenames

if __name__ == '__main__':


    rootdir = '/home/ssd/dataset/ShipRSImageNet/VOC_Format/Annotations/'
    text_dir = '/home/ssd/dataset/ShipRSImageNet/VOC_Format/ImageSets/'
    out_dir = '/home/zzn/Documents/zhangzhn_workspace/pycharm/ship_dataset/COCO_Format/'
    level_num = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    train_filenames = get_filenames(rootdir, text_dir, 'train')
    val_filenames = get_filenames(rootdir, text_dir, 'val')
    test_filenames = get_filenames(rootdir, text_dir, 'test')

    # print(train_filenames)
    # print('\n')


    train_json_file_name  = "{}ShipRSImageNet_bbox_train_level_{}.json".format(out_dir, level_num)
    val_json_file_name = "{}ShipRSImageNet_bbox_val_level_{}.json".format(out_dir, level_num)
    test_json_file_name = "{}ShipRSImageNet_bbox_test_level_{}.json".format(out_dir, level_num)

    ShipImageNet2COCOTrain(train_filenames, train_json_file_name, wordname_50, level_num)
    ShipImageNet2COCOTrain(val_filenames, val_json_file_name, wordname_50, level_num)
    ShipImageNet2COCOTest(test_filenames, test_json_file_name, wordname_50)

    print('Finished')


