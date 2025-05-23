import cv2
import glob
import os

input_dir = './person/images_raw/'
output_dir = './person/images_formatted/'

os.makedirs(output_dir, exist_ok=True)

for img_path in glob.glob(f'{input_dir}/*.jpg'):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (640, 480))
    filename = os.path.basename(img_path)
    cv2.imwrite(f'{output_dir}/{filename}', img_resized)