from rummySearch.tilesearch import TileDetector, order_points
import cv2
import imutils
import argparse
import numpy as np

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

if args["source"] == "webcam":
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
else:
    print("reading: " + args["source"])
    image = cv2.imread(args["source"])

while True:
    if args["source"] == "webcam":
        ret, image = cap.read()

    td.set_image(image, resize_width=960)

    tile_images = td.get_tile_images()

    for i, tile in enumerate(tile_images):
        cv2.imshow(str(i), tile)

    if args["source"] == "webcam":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

if args["source"] == "webcam":
    cap.release()

cv2.destroyAllWindows()
