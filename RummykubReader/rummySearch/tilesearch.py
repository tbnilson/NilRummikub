from typing import Any

import cv2
import imutils
import numpy as np


def order_points(points):
    pts = points.copy()
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)

    # Arrays of indices corresponding to the max and min sum of x+y for each point.
    # These can be at most length 2
    minindices = np.argwhere(s == np.amin(s))
    if len(minindices) > 1:
        if pts[minindices[0]][0][0] < pts[minindices[1]][0][0]:
            rect[0] = pts[minindices[0]]
            pts = np.delete(pts, minindices[0], axis=0)
        else:
            rect[0] = pts[minindices[1]]
            pts = np.delete(pts, minindices[1], axis=0)
    else:
        rect[0] = pts[np.argmin(s)]
        pts = np.delete(pts, np.argmin(s), axis=0)

    s = pts.sum(axis=1)
    maxindices = np.argwhere(s == np.amax(s))
    if len(maxindices) > 1:
        if pts[maxindices[0]][0][0] > pts[maxindices[1]][0][0]:
            rect[2] = pts[maxindices[0]]
            pts = np.delete(pts, maxindices[0], axis=0)
        else:
            rect[2] = pts[maxindices[1]]
            pts = np.delete(pts, maxindices[1], axis=0)
    else:
        rect[2] = pts[np.argmax(s)]
        pts = np.delete(pts, np.argmax(s), axis=0)

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


class TileDetector:
    image = None
    resize_ratio = 1

    def __init__(self):
        pass

    def detect(self, c, peri_frac):
        # c is the contour we want to analyze
        # peri_frac is the fraction of the perimiter we use for epsilon in approxPolyDP()
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri_frac * peri, True)
        return approx

    def classify(self, c, peri_frac=.04):
        approx = self.detect(c, peri_frac)
        shape = "unidentified"
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx == 4):
            shape = "quadrilateral"
        elif len(approx == 5):
            shape = "pentagon"
        else:
            shape = "circle"

        return shape, approx

    def set_image(self, image, resize_width=512):
        self.image = imutils.resize(image, width=resize_width)
        self.resize_ratio = image.shape[0] / float(self.image.shape[0])

    def get_threshold(self, threshold=0, kernel=9, image=None):
        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        if image is None:
            image = self.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel, threshold)
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def get_tile_contours(self) -> Any:
        thresh = self.get_threshold()
        thresh_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thresh_tiles = self.filter_tile_contours(thresh_cnts)
        thresh_tilemask = np.zeros(self.image.shape[:2], np.uint8)
        cv2.drawContours(thresh_tilemask, thresh_tiles, -1, 255, -1)
        img_masked = cv2.bitwise_and(self.image, self.image, mask=thresh_tilemask)
        return thresh_tiles, img_masked

    def filter_tile_contours(self, cnts):
        tiles = []
        img_area_frac = self.image.shape[0] * self.image.shape[1] * .005

        for c in cnts:
            poly = self.detect(c, 0.02)
            if len(poly) % 4 != 0 and len(poly) > 4:
                continue
            elif cv2.contourArea(poly) > img_area_frac:
                tiles.append(poly)

        return tiles

    def get_tile_images(self, width=200, height=300):
        thresh_tiles, _ = self.get_tile_contours()
        tile_rect = np.float32([[0, 0],
                                [width, 0],
                                [width, height],
                                [0, height]])
        tile_images = []
        for tile in thresh_tiles:
            tile_zeroangle = np.float32([
                [tile[0][0][0], tile[0][0][1]],
                [tile[1][0][0], tile[1][0][1]],
                [tile[2][0][0], tile[2][0][1]],
                [tile[3][0][0], tile[3][0][1]],
            ])

            if tile_zeroangle[0][0] == 0.0 and tile_zeroangle[0][1] == 0.0:
                break
            rect = order_points(tile_zeroangle)
            rect_rotated = np.roll(rect, 1, axis=0)

            matrix = cv2.getPerspectiveTransform(rect, tile_rect)
            tile_image_zerorotation = cv2.warpPerspective(self.image, matrix, (width, height))

            matrixr = cv2.getPerspectiveTransform(rect_rotated, tile_rect)
            tile_image_ninetyrotation = cv2.warpPerspective(self.image, matrixr, (width, height))

            gray_zero = cv2.cvtColor(tile_image_zerorotation, cv2.COLOR_BGR2GRAY)
            gray_ninety = cv2.cvtColor(tile_image_ninetyrotation, cv2.COLOR_BGR2GRAY)

            intensity_zero = self.get_threshold(image=tile_image_zerorotation)
            intensity_ninety = self.get_threshold(image=tile_image_ninetyrotation)

            print("First dim of gray image", len(gray_zero))

            zero_sum_high = np.sum(intensity_zero[:len(intensity_zero) // 2], dtype="int32")
            zero_sum_low = np.sum(intensity_zero[len(intensity_zero) // 2:], dtype="int32")

            ninety_sum_high = np.sum(intensity_ninety[:len(intensity_ninety) // 2], dtype="int32")
            ninety_sum_low = np.sum(intensity_ninety[len(intensity_ninety) // 2:], dtype="int32")
            print(ninety_sum_high - ninety_sum_low)
            print(zero_sum_high - zero_sum_low)
            if abs(zero_sum_high - zero_sum_low) > abs(ninety_sum_high - ninety_sum_low):
                tile_images.append(
                    intensity_zero if zero_sum_high < zero_sum_low else cv2.rotate(intensity_zero,
                                                                                   cv2.ROTATE_180))
            else:
                tile_images.append(
                    intensity_ninety if ninety_sum_high < ninety_sum_low else cv2.rotate(
                        intensity_ninety, cv2.ROTATE_180))
        return tile_images
