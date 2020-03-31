import zipfile
import tciaclient
import pandas as pd
import os
import matplotlib.pyplot as plt
import pydicom
import pathlib
import cv2
import numpy as np
import tqdm
import shutil

def getResponseString(response):
    if response.getcode() is not 200:
        raise ValueError("Server returned an error")
    else:
        return response.read()

def get_im_(outdir, uid):
    if not os.path.exists(outdir):
        pathlib.Path(outdir).mkdir(parents=True)
    response = client.get_image(uid)
    strResponseImage = getResponseString(response)
    with open("tmp.zip","wb") as fid:
        fid.write(strResponseImage)
        fid.close()
    fid = zipfile.ZipFile("tmp.zip")
    fid.extractall(outdir)
    fid.close()


api_key = "16ade9bc-f2fa-4a37-b357-36466a0020fc"
baseUrl="https://services.cancerimagingarchive.net/services/v3"
resource = "TCIA"

client = tciaclient.TCIAClient(api_key, baseUrl, resource)

im_type = 'cropped image file path'
started_files = False

data_sets = ['calc_case_description_train_set.csv',
             'mass_case_description_train_set.csv',
             'calc_case_description_test_set.csv',
             'mass_case_description_test_set.csv']

if not os.path.exists('data/train'): os.mkdir('data/train')
if not os.path.exists('data/test'): os.mkdir('data/test')
if not os.path.exists('data/train/0'): os.mkdir('data/train/0')
if not os.path.exists('data/train/1'): os.mkdir('data/train/1')
if not os.path.exists('data/test/0'): os.mkdir('data/test/0')
if not os.path.exists('data/test/1'): os.mkdir('data/test/1')

log = open('log', 'w')
label_map = {'MALIGNANT': 0, 'BENIGN': 1, 'BENIGN_WITHOUT_CALLBACK': 1}
for dataset_name in data_sets:
    dat_type = 'test' if 'test' in dataset_name else 'train'
    dataset = pd.read_csv(dataset_name)

    dataset[im_type + ' png'] = dataset[im_type].str.split('/').apply(lambda x: x[0]) + '.png'
    dataset['label'] = dataset['pathology'].map(label_map).astype(np.uint8)

    for i, (path, png_filename, label) in tqdm.tqdm(
            enumerate(zip(dataset[im_type], dataset[im_type + ' png'], dataset['label'])), total=len(dataset)):
        outdir = os.path.join('data', 'tmp')
        final_path = os.path.join('data', dat_type, str(label), png_filename)
        if os.path.exists(final_path):
            continue

        get_im_(outdir, path.split('/')[2])
        # assuming one CC per dir
        found = False
        for file in os.listdir(outdir):
            filename = os.path.join(outdir, file)

            # read the dcm file
            ds = pydicom.read_file(filename)

            # validate type
            if not hasattr(ds, 'SeriesDescription'):
                continue
            if ds.SeriesDescription.lower() != 'cropped images':
                os.remove(filename)  # ROI mask - handle seperatly
                continue

            found = True
            # save the image as png wit hapropriot path
            cv2.imwrite(os.path.join('data', dat_type, str(label), png_filename), ds.pixel_array)
        shutil.rmtree(outdir)
        if not found:  # alert and I'll look into it someday...
            log.write(str(i))
            print(i)
        #assert (found)  # all files must be found
    dataset.to_csv('new_' + dataset_name)


