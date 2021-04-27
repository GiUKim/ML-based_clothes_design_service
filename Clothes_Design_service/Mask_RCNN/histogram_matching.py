import cv2
import matplotlib.pyplot as plt
import argparse
import os
from skimage import exposure
IMAGE_DIR = os.path.abspath("C:\\Users\\123\\Desktop\\Clothes_Design\\public\\gan_output")
MATCH_DIR = os.path.abspath("C:\\Users\\123\\Desktop\\Clothes_Design\\public\\match_output")
if __name__== "__main__":
    ref_name = os.path.join(IMAGE_DIR, "1-1.jpg")
    ref = cv2.imread(ref_name, cv2.IMREAD_COLOR)
    for i in range(2, 7):
        string = str(1) + "-" + str(i) + ".jpg"
        string = os.path.join(IMAGE_DIR, string)
        src = cv2.imread(string, cv2.IMREAD_COLOR)
        print("[INFO] performing histogram matching...")
        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src, ref, multichannel=multi)
        # cv2.imshow("Source", src)
        # cv2.imshow("Reference", ref)
        # cv2.imshow("Matched", matched)
        # cv2.waitKey(0)

    tgt = []
    for i in range(2, 7):
        string = str(1) + "-" + str(i) + ".jpg"
        string = os.path.join(IMAGE_DIR, string)
        tgt.append(cv2.imread(string, cv2.IMREAD_COLOR))
        tgt[i - 2] = cv2.resize(tgt[i - 2], dsize=(256,256), interpolation=cv2.INTER_LINEAR)

    for i in range(2, 7):
        matched = "match"+str(i-2)+".jpg"
        matched_name = os.path.join(MATCH_DIR, matched)
        # save match1 ~ match5 image file
        cv2.imwrite(matched_name, tgt[i - 2])
