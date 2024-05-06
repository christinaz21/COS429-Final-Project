import cv2
import numpy as np
import os
import pandas as pd


def calculate_size(mask_path):
    mask = cv2.imread(mask_path)
    #print(mask)
    mask_pixels = np.sum(mask) / 255
    #print(mask_pixels)
    #mask_pixels = cv2.countNonZero(mask)

    height, width, channels = mask.shape
    total_pixels = height * width * channels
    #print(channels)
    prop = mask_pixels / total_pixels
    #print(prop)
    #print(mask_pixels)
    #print(mask.size)
    return prop


def get_mask_sizes(mask_folder):

    prop_dict = {}
    prop_dict_cat = {}

    for dirpath, dirnames, filenames in os.walk(mask_folder):
        current_folder = os.path.basename(dirpath)
        #print(current_folder)
        if current_folder == mask_folder: 
            continue

        files = [f for f in filenames if not f.startswith('.DS_Store')]
        
        cat_sizes_list = []
        for filename in files:

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
             

                try:

                    prop = calculate_size(image_path)
                    prop_dict[image_path] = prop
                    cat_sizes_list.append(prop)

                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")

        cat_sizes = np.array(cat_sizes_list)
        avg_size = np.mean(cat_sizes)
        prop_dict_cat[current_folder] = avg_size
        
    mask_props = pd.DataFrame(list(prop_dict.items()), columns=['ImagePath', 'Proportion'])
    cat_props = pd.DataFrame(list(prop_dict_cat.items()), columns=['Category', 'Average Proportion'])

    return cat_props, mask_props


# change input and output folder for each scene
mask_folder = 'masks'
cat_props, mask_props = get_mask_sizes(mask_folder)


# Create a Pandas Excel writer using openpyxl as the engine
with pd.ExcelWriter('sizes_by_cat.xlsx', engine='openpyxl') as writer:
    cat_props.to_excel(writer, sheet_name='Avg prop per category', index=False)
    mask_props.to_excel(writer, sheet_name='Prop per image', index=False)


