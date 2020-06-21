import cv2
import imutils


class TileDetector:
    image = None
    resize_ratio = 1
    thresh = None

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

    def get_threshold(self, threshold, kernel=5):
        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        self.thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
        return self.thresh

    def get_masked_img(self, lower, upper):
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower, upper)
        result = cv2.bitwise_and(self.image, self.image, mask=mask)
        return result, mask

    def find_tiles(self, cnts):
        tiles = []
        img_area_frac = self.image.shape[0] * self.image.shape[1] * .005

        for c in cnts:
            poly = self.detect(c, 0.04)
            if len(poly)==4:
                if cv2.contourArea(poly) > img_area_frac:
                    tiles.append(poly)

        return tiles

