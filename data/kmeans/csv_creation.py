import glob
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def make_csv_imageNet(path_dir: str, path_file_name: str, output_path: str):
    files = []
    classes_id = []
    with open(path_file_name) as f:
        lines = f.readlines()
        i = 0
        for cls in lines:
            cls_path = os.path.join(path_dir, cls.split()[0])
            # print(cls_path)
            classes_id.append(cls.split('/')[0])
            # print((glob.glob(data_path + '/*')))
            files.append(cls_path.split()[0] + ".JPEG")
            i += 1
    # print(i)
    print(len(files))
    print(files[-1])
    print(len(classes_id))
    dict = {'path': files, 'class': classes_id}
    df = pd.DataFrame(dict)
    print(len(df))
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    np.save('label_encoder_classes.npy', le.classes_)
    print(len(df))
    df.to_csv(output_path, index=False)


def make_csv_imageNet_val(path_dir: str, output_path: str):
    files = []
    classes_id = []
    list_dir = os.listdir(path_dir)
    for dir in list_dir:
        # print(dir)
        i = 0
        abs_path_dir = os.path.join(path_dir, dir)
        for file in os.listdir(abs_path_dir):
            classes_id.append(dir)
            files.append(os.path.join(abs_path_dir, file))
            i += 1
    # print(i)
    print(len(files))
    print(files[-1])
    print(len(classes_id))
    dict = {'path': files, 'class': classes_id}
    df = pd.DataFrame(dict)
    print(len(df))
    le = LabelEncoder()
    le.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
    df['class'] = le.transform(df['class'])
    print(len(df))
    df.to_csv(output_path, index=False)


def make_csv_city(txt_file: str, raw_img_path: str, annotation_path: str):
    list_raw = []
    list_ann = []
    with open(txt_file) as f:
        lines = f.readlines()
    for image in lines:
        image = image.rstrip('\n')
        list_raw.append(os.path.join(raw_img_path, image + "_leftImg8bit.png"))
        list_ann.append(os.path.join(annotation_path, image + "_gtFine_labelTrainIds.png"))
    dict = {'path': list_raw, 'class': list_ann}
    df = pd.DataFrame(dict)
    df.to_csv("./city_train.csv", index=False)


def make_csv_ADE(raw_img_path: str, additional_saving_string=""):
    list_raw = []
    list_ann = []
    for image in os.listdir(os.path.join(raw_img_path, "images", additional_saving_string)):
        list_raw.append(os.path.join("images", additional_saving_string, image))
        list_ann.append(os.path.join("annotations", additional_saving_string, image.split(".")[0] + ".png"))
    dict = {'path': list_raw, 'annotations': list_ann}
    df = pd.DataFrame(dict)
    df.to_csv(f"./ADE_{additional_saving_string}.csv", index=False)
