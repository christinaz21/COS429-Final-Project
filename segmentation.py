import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# def apply_mask_rcnn(image):
#     model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
#     model.eval()
#     transform = F.to_tensor(image).unsqueeze(0)
#     with torch.no_grad():
#         prediction = model(transform)
#     return prediction[0]
    
def apply_mask_rcnn(image):
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    model.eval()
    transform = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(transform)

    labels = prediction[0]['labels'].cpu().numpy()

    coco_ids_people_animals = [1, 3, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27] # COCO ids for people and animals
    masks = prediction[0]['masks'][np.isin(labels, coco_ids_people_animals)] > 0.5  # only keep the masks for people and animals
    
    return masks

def apply_grabcut(image, mask):
    #for i in range(mask.shape[0]):
       # for j in range(mask.shape[1]):
            #print(mask[i, j])
    #plt.imshow(image)
    #plt.plot(mask)
    #plt.show()
    mask[mask == 1] = cv2.GC_FGD  # Foreground
    mask[mask == 0] = cv2.GC_BGD  # Background
    #print(mask)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    #plt.imshow(mask)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

def segment_image(image_path):
    image = load_image(image_path)
    masks = apply_mask_rcnn(image)
    #print(outputs)
    # masks = outputs['masks'] > 0.5
    mask_combined = np.zeros(masks.shape[-2:], dtype=np.uint8)
    if masks.numel() > 0:
        mask_combined = masks[0, 0].byte().cpu().numpy()
    #plt.imshow(mask_combined)
    segmented = apply_grabcut(image, mask_combined)
    #plt.imshow(segmented)
    return image, segmented

def process_images(input_folder, output_folder, mask_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    
    count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            try:
                original_image, segmented_mask = segment_image(image_path)
                background_image = original_image.copy()
                background_image[segmented_mask] = 0
                cv2.imwrite(output_path, cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR))

                # Get the dimensions of the image
                height, width, channels = original_image.shape
                # Create a black image of the same dimensions
                black_image = np.zeros((height, width, channels), dtype=np.uint8)
                black_image[segmented_mask] = 1
                black_image = black_image.astype(np.uint8) * 255 
                cv2.imwrite(mask_path, black_image)
                #print(f"Processed and saved background image to {output_path}")
                count += 1
                if count % 50 == 0:
                    print("processed ", count)
                
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")

# change input and output folder for each scene
input_folder = 'images'
output_folder = 'processed'
mask_folder = 'masks'
process_images(input_folder, output_folder, mask_folder)
