import cv2
import pytesseract as pytesseract
from PIL import Image
from skimage import exposure
import imutils as imutils
import numpy as np
import datetime
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# https://pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

path = r'../images/'
filename = 'test2.jpg'

start = datetime.datetime.now()
image = cv2.imread(path + filename)

ratio = image.shape[0] / 300.0
orig = image.copy()
image = imutils.resize(image, height=300)

edges = cv2.Canny(cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17), 30, 200)

# plt.subplot(121), plt.imshow(image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

contours = sorted(imutils.grab_contours(cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)),
                  key=cv2.contourArea, reverse=True)[:10]
boardContour = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    if len(approx) == 4:
        boardContour = approx
        break

# cv2.drawContours(image, [boardContour], -1, (0, 255, 0), 3)
# cv2.imshow("Game Boy Screen", image)
# cv2.waitKey(0)

points = boardContour.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")

s = points.sum(axis=1)
rect[0] = points[np.argmin(s)]
rect[2] = points[np.argmax(s)]

diff = np.diff(points, axis=1)
rect[1] = points[np.argmin(diff)]
rect[3] = points[np.argmax(diff)]
rect *= ratio

(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

ocr_image = cv2.cvtColor(cv2.warpPerspective(orig, cv2.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight)),
                         cv2.COLOR_BGR2GRAY)

# save the cropped image to file
# ocr_image = np.array(Image.fromarray(warp.astype(np.uint8)))
# ocr_image = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
ocr_image, img_bin = cv2.threshold(ocr_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ocr_image = cv2.bitwise_not(img_bin)

kernel = np.ones((2, 1), np.uint8)
ocr_image = cv2.erode(ocr_image, kernel, iterations=1)
ocr_image = cv2.dilate(ocr_image, kernel, iterations=1)

# crop outside borders
ocr_image = ocr_image[int(len(ocr_image) * 0.015):-int(len(ocr_image) * 0.015),
            int(len(ocr_image[0]) * 0.015):-int(len(ocr_image[0]) * 0.015)]
plt.imshow(ocr_image)
plt.show()
box_height = len(ocr_image) // 9
box_width = len(ocr_image[0]) // 9
print(box_height, box_width)
print(ocr_image.shape)
# for r in range(9):
#     for c in range(9):
#         # print(r, c)
#         box_start_x = c * box_width
#         box_start_y = r * box_height
#         box = ocr_image[box_start_y + int(0.15 * box_height):box_start_y + box_height - int(0.15 * box_height),
#               box_start_x + int(0.15 * box_width):box_start_x + box_width - int(0.15 * box_width)]
#         print(box.shape)
#         plt.imshow(box)
#         plt.show()
#         text = pytesseract.image_to_string(box, config='--psm 7')
#         print(text)

# iterate through sub-boxes:
sub_box_width = len(ocr_image[0]) // 3
sub_box_height = len(ocr_image) // 3
sub_box_padding_percent = 0.02
sub_box_padding_x = int(sub_box_width * sub_box_padding_percent)
sub_box_padding_y = int(sub_box_height * sub_box_padding_percent)

for R in range(3):
    for C in range(3):
        sub_box_start_x = C * sub_box_width
        sub_box_start_y = R * sub_box_height
        sub_box_padding_left = 0
        sub_box_padding_right = 0
        sub_box_padding_up = 0
        sub_box_padding_down = 0

        sub_box = ocr_image[
                  sub_box_start_y + sub_box_padding_y:sub_box_start_y + sub_box_height - sub_box_padding_y,
                  sub_box_start_x + sub_box_padding_x:sub_box_start_x + sub_box_width - sub_box_padding_x]

        current_sub_box_width = len(sub_box[0])
        current_sub_box_height = len(sub_box)
        sub_box_center_x = current_sub_box_width // 2
        sub_box_center_y = current_sub_box_height // 2

        # sub_box = ocr_image[
        #           sub_box_start_y:sub_box_start_y + sub_box_height,
        #           sub_box_start_x:sub_box_start_x + sub_box_width]

        for x in range(current_sub_box_width):
            if sub_box[sub_box_center_y, x] < 50:
                sub_box_padding_left = x
                break

        for x in reversed(range(current_sub_box_width)):
            if sub_box[sub_box_center_y, x] < 50:
                sub_box_padding_right = x
                break

        for x in range(current_sub_box_height):
            if sub_box[x, sub_box_center_x] < 50:
                sub_box_padding_up = x
                break

        for x in reversed(range(current_sub_box_height)):
            if sub_box[x, sub_box_center_x] < 50:
                sub_box_padding_down = x
                break

        print(sub_box_padding_left, sub_box_padding_right, sub_box_padding_up, sub_box_padding_down)

        sub_box = sub_box[sub_box_padding_up:sub_box_padding_down, sub_box_padding_left:sub_box_padding_right]

        plt.imshow(sub_box)
        plt.show()

# cv2.imwrite("warped.png", warp)
print(datetime.datetime.now() - start)
