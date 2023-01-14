import easyocr
import cv2
import numpy as np

img = cv2.imread('5.jpg', 0)        # open image in gray scale

# invert = 255 - img                # invert color

increase = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# thresholding
# _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
# _, otsu = cv2.threshold(increase, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# adaptive_average = cv2.adaptiveThreshold(increase, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
# adaptive_gaussian = cv2.adaptiveThreshold(increase, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

# noise removal outside
# erosion = cv2.erode(adaptive_gaussian, np.ones((5, 5), np.uint8))
# opening = cv2.dilate(erosion, np.ones((5, 5), np.uint8))

# noise removal inside
# dilation = cv2.dilate(gray, np.ones((5, 5), np.uint8))
# closing = cv2.erode(dilation, np.ones((5, 5), np.uint8))

# show image
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# imS = cv2.resize(increase, (1920, 1280))  # Resize image
# cv2.imshow("output", imS)
# cv2.waitKey(0)


reader = easyocr.Reader(['bg'], gpu=False)
result = reader.readtext(increase, detail=0, paragraph=True, contrast_ths=0.05, adjust_contrast=0.7, add_margin=0.25,
                         width_ths=0.7, decoder='beamsearch')

print(result)
