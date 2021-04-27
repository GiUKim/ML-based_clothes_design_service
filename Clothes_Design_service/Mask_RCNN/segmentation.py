import numpy as np
import skimage.draw
import os
import argparse
import cv2
import datetime
import sys
IMAGE_DIR = os.path.abspath("C:\\Users\\123\\Desktop\\Clothes_Design\\public\\match_output")
# C:\Users\123\Desktop\Clothes_Design\public\mask_output
MASK_DIR = os.path.abspath("C:\\Users\\123\\Desktop\\Clothes_Design\\public\\mask_output")
OUTPUT_DIR = os.path.abspath("C:\\Users\\123\\Desktop\\Clothes_Design\\public\\output")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='edit result image')
    # parser.add_argument('--mask', required=False,
    #                     metavar="path or URL to image"
    #                   )
    # parser.add_argument('--gan', required=False,
    #                     metavar="path or URL to image")

    # args = parser.parse_args()
    # mask_image = os.path.join(ROOT_DIR, args.mask)
    # gan_image = os.path.join(ROOT_DIR, args.gan)
    mask_image = "mask.jpg"
    mask_image = os.path.join(MASK_DIR, mask_image)
    mask_image = cv2.imread(mask_image, cv2.IMREAD_COLOR)
    h, w, c = mask_image.shape
    print(h, w)
    file_name = "{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    for i in range(0, 5):
        gan_image = "match" + str(i) + ".jpg"
        gan_image = os.path.join(IMAGE_DIR, gan_image)
        gan_image = cv2.imread(gan_image, cv2.IMREAD_COLOR)
        gan_image = cv2.resize(gan_image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        kernel1d = cv2.getGaussianKernel(5, 0.5)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())
        revised_gan = cv2.filter2D(gan_image, -1, kernel2d)  # convolve
        splash = np.where(mask_image==255, revised_gan, 0)
        output_filename = file_name+"_"+str(i+1)+".jpg"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_filepath, splash)
        print(file_name + "_" + str(i+1) +".jpg")
    #revised_gan = cv2.resize(gan_image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
    
    

    

    # cv2.imshow("src", splash)
    # cv2.waitKey(0)


