import cv2
from skimage import exposure
import imutils as imutils
import numpy as np
import datetime
from matplotlib import pyplot as plt

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

warp = exposure.rescale_intensity(
    cv2.cvtColor(cv2.warpPerspective(orig, cv2.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight)),
                 cv2.COLOR_BGR2GRAY),
    out_range=(0, 255))

# save the cropped image to file


cv2.imwrite("warped.png", warp)
print(datetime.datetime.now()-start)


