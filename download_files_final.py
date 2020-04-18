# python 2
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

# image file path, cropped image file path, ROI mask file pat
im_type = 'ROI mask file path'
dir_name = 'ROI_file'
dir_name_full = 'data/' + dir_name

# 'full mammogram images', 'cropped images', 'roi mask images'
series_description = 'roi mask images'

csv_dir = 'csv_files'
data_sets = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]

if not os.path.exists(dir_name_full): os.mkdir(dir_name_full)
if not os.path.exists(dir_name_full+'/train'): os.mkdir(dir_name_full+'/train')
if not os.path.exists(dir_name_full+'/test'): os.mkdir(dir_name_full+'/test')
if not os.path.exists(dir_name_full+'/train/0'): os.mkdir(dir_name_full+'/train/0')
if not os.path.exists(dir_name_full+'/train/1'): os.mkdir(dir_name_full+'/train/1')
if not os.path.exists(dir_name_full+'/test/0'): os.mkdir(dir_name_full+'/test/0')
if not os.path.exists(dir_name_full+'/test/1'): os.mkdir(dir_name_full+'/test/1')

log = open('log', 'w')
label_map = {'MALIGNANT': 0, 'BENIGN': 1, 'BENIGN_WITHOUT_CALLBACK': 1}
prob = ['Calc-Training_P_00474_LEFT_MLO_1.png']
for dataset_name in data_sets:
    dat_type = 'test' if 'test' in dataset_name else 'train'
    dataset = pd.read_csv(dataset_name)

    dataset[im_type + ' png'] = dataset[im_type].str.split('/').apply(lambda x: x[0]) + '.png'
    dataset['label'] = dataset['pathology'].map(label_map).astype(np.uint8)

    for i, (path, png_filename, label) in tqdm.tqdm(
            enumerate(zip(dataset[im_type], dataset[im_type + ' png'], dataset['label'])), total=len(dataset)):
        outdir = os.path.join('data', 'tmp')
        final_path = os.path.join('data', dir_name, dat_type, str(label), png_filename)
        if os.path.exists(final_path) and png_filename not in prob:
            continue

        get_im_(outdir, path.split('/')[2])
        # assuming one CC per dir
        found = False
        dir_size = len(os.listdir(outdir))
        for file in os.listdir(outdir):
            # find bigger file

            filename = os.path.join(outdir, file)

            # read the dcm file
            ds = pydicom.read_file(filename)
            w, h = ds.pixel_array.shape

            # validate type
            if w>2000 and h>2000:
                found = True
                cv2.imwrite(final_path, ds.pixel_array)
                break
            elif dir_size > 1:
                # might not be it then
                if not hasattr(ds, 'SeriesDescription'):
                    continue
                if ds.SeriesDescription.lower() != series_description:
                    os.remove(filename)  # ROI mask - handle seperatly
                    continue

            found = True
            # save the image as png wit hapropriot path
            cv2.imwrite(final_path, ds.pixel_array)
        if not found:  # alert and I'll look into it someday...
            log.write(str(i))
            print(i)
        shutil.rmtree(outdir)
        #assert (found)  # all files must be found
    dataset.to_csv(dataset_name)


