import os
import cv2
import glob
# smaple the [512,512] sample to be the size of [96, 96]
# Paths
source_dir = "scratch/sourcecode/cem-dataset/benchdata/guay/2d/train/masks"
target_dir = "/home/codee/scratch/sourcecode/cem-dataset/benchdata/guay/2d/subtrain/masks" 

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Function to crop the central region of the image
def crop_center(img, crop_size):
    y, x, _ = img.shape
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)    
    return img[starty:starty+crop_size, startx:startx+crop_size]

# Process each image
for img_path in glob.glob(os.path.join(source_dir, "*.tiff")):
    img = cv2.imread(img_path)
    cropped_img = crop_center(img, 96)
    print(len(cropped_img))
    base_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(target_dir, base_name), cropped_img)

print("Processing complete.")
