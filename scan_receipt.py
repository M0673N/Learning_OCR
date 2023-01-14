import easyocr
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt


def show_img(img):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def sort_points(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)

    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    dif = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(dif)]
    new_points[3] = points[np.argmax(dif)]

    return new_points


def find_contours(img):  # EXTERNAL, RETR_TREE
    conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:6]
    return conts


def transform_image(image_file):
    larger = None
    img = cv2.imread(image_file)
    original = img.copy()
    (H, W) = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 60, 160)
    conts = find_contours(edged.copy())
    for c in conts:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(aprox) == 4:
            larger = aprox
            break

    if larger is None:
        return img

    cv2.drawContours(img, larger, -1, (120, 255, 0), 28)
    cv2.drawContours(img, [larger], -1, (120, 255, 0), 2)

    points_larger = sort_points(larger)
    pts1 = np.float32(points_larger)
    pts2 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transform = cv2.warpPerspective(original, matrix, (W, H))

    return transform


def adjust_brightness(img):
    brightness = 50
    contrast = 80
    adjust = np.int16(img)
    adjust = adjust * (contrast / 127 + 1) - contrast + brightness
    adjust = np.clip(adjust, 0, 255)
    adjust = np.uint8(adjust)
    return adjust


def remove_edges(img):
    margin = 18
    (H, W) = img.shape[:2]
    img_edges = img[margin:H - margin, margin:W - margin]
    return img_edges


def process_img(img):
    processed_img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    show_img(processed_img)
    processed_img = adjust_brightness(processed_img)
    show_img(processed_img)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    show_img(processed_img)
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    show_img(processed_img)
    return processed_img


img = transform_image('123.jpg')
show_img(img)
img = process_img(img)

reader = easyocr.Reader(['bg'], gpu=False)
result = reader.readtext(img, detail=0, paragraph=True, contrast_ths=0.05, adjust_contrast=0.7, add_margin=0.25,
                         width_ths=0.7, decoder='beamsearch')

print(result)
