from rummySearch.tilesearch import TileDetector
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
cv2.namedWindow(TRACKBARS)
cv2.resizeWindow(TRACKBARS, 640, 300)
cv2.createTrackbar("Hue Min", TRACKBARS, 20, 179, nothing)
cv2.createTrackbar("Hue Max", TRACKBARS, 69, 179, nothing)
cv2.createTrackbar("Sat Min", TRACKBARS, 0, 255, nothing)
cv2.createTrackbar("Sat Max", TRACKBARS, 94, 255, nothing)
cv2.createTrackbar("Val Min", TRACKBARS, 151, 255, nothing)
cv2.createTrackbar("Val Max", TRACKBARS, 255, 255, nothing)

if args["source"] == "webcam":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
else:
    image = cv2.imread(args["source"])

while True:
    if args["source"] == "webcam":
        ret, image = cap.read()

    td.set_image(image, resize_width=960)

    h_min = cv2.getTrackbarPos("Hue Min", TRACKBARS)
    h_max = cv2.getTrackbarPos("Hue Max", TRACKBARS)
    s_min = cv2.getTrackbarPos("Sat Min", TRACKBARS)
    s_max = cv2.getTrackbarPos("Sat Max", TRACKBARS)
    v_min = cv2.getTrackbarPos("Val Min", TRACKBARS)
    v_max = cv2.getTrackbarPos("Val Max", TRACKBARS)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    img_masked, mask = td.get_masked_img(lower, upper)

    cnts, heirarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = td.find_tiles(cnts)

    cv2.drawContours(img_masked, tiles, -1, (0, 255, 0), 3)
    cv2.imshow("mask", img_masked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
