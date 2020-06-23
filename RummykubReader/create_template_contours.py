from rummySearch.tilesearch import TileDetector, order_points
import cv2
import imutils
import argparse
import numpy as np

td = TileDetector()

image = cv2.imread("rummySearch/all_black_tiles.jpg")

td.set_image(image, resize_width=960)

tile_images = td.get_tile_images()

for i, tile in enumerate(tile_images):
    cv2.imshow(str(i), tile)

cv2.waitKey(0)
