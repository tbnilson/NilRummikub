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
cv2.namedWindow(TRACKBARS)
cv2.resizeWindow(TRACKBARS, 640, 300)
cv2.createTrackbar("Hue Min", TRACKBARS, 20, 179, nothing)
cv2.createTrackbar("Hue Max", TRACKBARS, 69, 179, nothing)
cv2.createTrackbar("Sat Min", TRACKBARS, 0, 255, nothing)
cv2.createTrackbar("Sat Max", TRACKBARS, 94, 255, nothing)
cv2.createTrackbar("Val Min", TRACKBARS, 151, 255, nothing)
cv2.createTrackbar("Val Max", TRACKBARS, 255, 255, nothing)
cv2.createTrackbar("threshold", TRACKBARS, 1, 100, nothing)

if args["source"] == "webcam":
    cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
else:
    print("reading: " + args["source"])
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

    # cnts, heirarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # tiles = td.find_tiles(cnts)
    #
    # tilemask = np.zeros(td.image.shape[:2], np.uint8)
    # cv2.drawContours(tilemask, tiles, -1, 255, -1)
    # img_masked = cv2.bitwise_and(td.image, td.image, mask=tilemask)

    # cv2.imshow("mask", td.image)

    thresh = td.get_threshold()
    thresh_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh_tiles = td.find_tiles(thresh_cnts)
    thresh_tilemask = np.zeros(td.image.shape[:2], np.uint8)
    cv2.drawContours(thresh_tilemask, thresh_tiles, -1, 255, -1)
    img_masked = cv2.bitwise_and(td.image, td.image, mask=thresh_tilemask)
    cv2.imshow("adaptive threshold", img_masked)

    tile_width = 100
    tile_height = 150
    tile_rect = order_points(np.float32([[0, 0], [0, tile_height], [tile_width, tile_height], [tile_width, 0]]))
    tile_images = []
    for tile in thresh_tiles:
        tile32 = np.float32([
            [tile[0][0][0], tile[0][0][1]],
            [tile[1][0][0], tile[1][0][1]],
            [tile[2][0][0], tile[2][0][1]],
            [tile[3][0][0], tile[3][0][1]],
        ])

        if tile32[0][0] == 0.0 and tile32[0][1] == 0.0:
            tile_images.append(np.zeros(tile_rect.shape))
            break
        rect = order_points(tile32)
        print("unorderd")
        print(tile32)
        print("ordered")
        print(rect)
        matrix = cv2.getPerspectiveTransform(rect, tile_rect)
        tile_images.append(cv2.perspectiveTransform(td.image, matrix))

    if len(tile_images)>0:
        cv2.imshow("warped", tile_images[0])

    if args["source"] == "webcam":
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

if args["source"] == "webcam":
    cap.release()
cv2.destroyAllWindows()
