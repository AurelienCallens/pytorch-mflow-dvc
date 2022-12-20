import os
import glob
from torchvision.io import read_image

def verify_channel_number(data_repo):
    list_images = glob.glob(data_repo, recursive=True)

    for image in list_images:
        img = read_image(image)
        if img.shape[0] != 3:
            os.remove(img)
            print('Removed {image_path} with {img_channel} channels'.format(image_path=image, img_channel=img.shape[0]))

    print("All the images have been verified!")

if __name__ == '__main__':
    verify_channel_number(data_repo='./data/raw/**/*.jpg')