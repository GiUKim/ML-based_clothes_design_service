import cv2
import matplotlib.pyplot as plt
import argparse
from skimage import exposure

if __name__== "__main__":
    for i in range(1,4):
        string = "target" + str(i) + ".jpg"
        src = cv2.imread(string, cv2.IMREAD_COLOR)
        ref = cv2.imread("origin.jpg", cv2.IMREAD_COLOR)
        # determine if we are performing multichannel histogram matching
        # and then perform histogram matching itself
        print("[INFO] performing histogram matching...")
        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src, ref, multichannel=multi)
        # show the output images
        cv2.imshow("Source", src)
        cv2.imshow("Reference", ref)
        cv2.imshow("Matched", matched)
        cv2.waitKey(0)

    tgt = []
    for i in range(1, 4):
        string = "target" + str(i) + ".jpg"
        tgt.append(cv2.imread(string, cv2.IMREAD_COLOR))
        tgt[i - 1] = cv2.resize(tgt[i - 1], dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

    for i in range(1, 4):
        string = "target" + str(i) + ".jpg"
        cv2.imwrite(string, tgt[i - 1])

    dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.imshow("dst2", dst2)



