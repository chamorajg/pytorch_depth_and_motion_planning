import os
import glob
import json
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm

def filter_images():
    """ This function gives filtered frame paths of every video in the format 
        of json stored in filtered_image.json"""
    base_directory = './images'
    data_dictionary = {}
    directories = image_files = sorted(glob.glob('{}/*/'.format(base_directory)))
    idx = 0
    pbar = tqdm(total=len(image_files), desc="Filtering Images", position=0)
    for dir in directories:
        image_files = sorted(glob.glob('{}/*.jpg'.format(dir)))
        data_dictionary[idx] = image_files[0]
        idx += 1
        for i in range(0, len(image_files)-1):
            for j in range(i+1, len(image_files)):
                if np.greater( np.mean( np.abs( np.array(Image.open(image_files[j])) - 
                        np.array(Image.open(image_files[i])) ) ), 0.05):
                    data_dictionary[idx] = image_files[j]
                    idx += 1
                    pbar.update(1)
                    i = j
                    break
    json.dump( data_dictionary, open('./filtered_image.json','w') )

if __name__ == '__main__':
    filter_images()