import time

from rummySearch.tilesearch import TileDetector, order_points
import cv2
import imutils
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TRACKBARS = "Trackbars"


def nothing(x):
    pass


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

td = TileDetector()
cap = None
image = None

plt.ion()
fig, axes = plt.subplots(nrows=2, ncols=5, gridspec_kw={'wspace': 0.05, 'hspace': 0})

emptyim = np.ones([300, 200], dtype=np.uint8)
emptyim.fill(255)
ax_images = [None] * len(axes.flatten())

for num, ax in enumerate(axes.flatten()):
    ax.axis('off')
    im = ax.imshow(emptyim)
    im.norm.vmin = 0
    im.norm.vmax = 255
    ax_images[num] = im

if args["source"] == "webcam":
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
else:
    print("reading: " + args["source"])
    image = cv2.imread(args["source"])

first_loop = True
while True:
    tic = time.time()
    if args["source"] == "webcam":
        ret, image = cap.read()

    td.set_image(image, resize_width=960)
    cv2.imshow("Original Image", td.image)
    tile_images = td.get_tile_images()
    # print(len(tile_images))

    for i, ax_image in enumerate(ax_images):

        if i < len(tile_images):
            tile = tile_images[i]
        else:
            tile = emptyim

        im = ax_images[i]
        im.set_data(tile)
        plt.pause(.05)

    toc = time.time()
    print(toc-tic)
    if args["source"] == "webcam":
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

if args["source"] == "webcam":
    cap.release()

cv2.destroyAllWindows()
