import numpy as np
from cv2 import cv2 as cv
import sys


def get_img():
    hex_2_bgr = lambda h: [int(h[i:i + 2], 16) for i in (0, 2, 4)]

    input = sys.stdin.read
    data = []
    for _ in range(1024):
        data.append(list(map(hex_2_bgr, input().split())))

    matrix = np.matrix(data)
    img = cv.imencode(matrix, cv.IMREAD_COLOR)
    return img


def get_img_from_file():
    hex_2_bgr = lambda h: np.array(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    array = np.arange(0, 1280 * 1024, 1, np.uint8)
    array = np.reshape(array, (1280, 1024))

    file = open('0', mode='r')

    data = []
    for i in range(1024):
        sth = file.readline().split()
        data.append(list(map(hex_2_bgr, sth)))

    print(data[0][0])
    data = np.array(data)
    matrix = np.reshape(data, (1280, 1024, 3))
    # print(matrix)
    # img = cv.imencode(data, cv.IMREAD_COLOR)
    return matrix


def main(debug=False):
    if debug:
        from matplotlib import pyplot as plt

    answer = 0

    if debug:
        img = cv.imread('image.jpg')
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        print(img[0, 0], type(img[0, 0]))
    else:
        img = get_img_from_file()

    hsv_min = np.array((100, 40, 56), np.uint8)
    hsv_max = np.array((153, 255, 255), np.uint8)
    color_red = (0, 0, 255)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv_img, hsv_min, hsv_max)

    # grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # ret, thresh = cv.threshold(grey, 125, 255, cv.THRESH_BINARY)
    img2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if debug:
        cv.drawContours(hsv_img, contours, -1, (255, 255, 255), 2)

        plt.subplot(121), plt.imshow(hsv_img, cmap='gray')
        plt.title('gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img2, cmap='gray')
        plt.title('contours'), plt.xticks([]), plt.yticks([])
        plt.show()

    for i in range(len(contours)):
        try:
            cnt = contours[i]
            M = cv.moments(cnt)

            cx = int(M['m10'] / M['m00'])  # центр контура
            cy = int(M['m01'] / M['m00'])
            dArea = M['m00']

            color = list(img[cy, cx])
            if debug:
                print(color, [cx, cy])

            if dArea >= 180:  # красный
                answer += 1

        except Exception:
            if debug:
                print(Exception)

    if debug:
        print(f'answer is {answer}')
    else:
        print(answer)


if __name__ == '__main__':
    main(debug=False)
