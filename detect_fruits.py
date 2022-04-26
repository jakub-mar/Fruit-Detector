import cv2
import os
import numpy as np
from pykuwahara import kuwahara
import json


def ReadyImage(source, interpolation_type, scale):
    # Reading image from file and resizing it
    image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    image_resized = cv2.resize(
        image, (int(image.shape[0]/scale), int(image.shape[1]/scale)), interpolation=interpolation_type)
    return image_resized


def ImageFiltering(if_image):
    # Apllying filters to transform the image
    image_kuwahara = kuwahara(if_image, method='mean', radius=15)
    image_median = cv2.medianBlur(image_kuwahara, 15)
    image_hsv = cv2.cvtColor(image_median, cv2.COLOR_BGR2HSV)
    return image_hsv


def FruitsDetection(filtered_image):
    # Apllying a special mask that cut out every fruit from image
    fruits_l = (0, 94, 0)
    fruits_h = (255, 255, 255)

    fruits_mask = cv2.inRange(filtered_image, fruits_l, fruits_h)
    opening = cv2.morphologyEx(
        fruits_mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    closing = cv2.morphologyEx(
        opening, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    return closing


def FindBanana(filtered_image, fruits_detected, area, image_to_contours):
    # Cutting out bananas by specific thresholds
    banana_counter = 0

    banana_l = (20, 59, 180)
    banana_h = (69, 255, 255)
    banana_mask = cv2.inRange(filtered_image, banana_l, banana_h)
    banana_mask = cv2.morphologyEx(
        banana_mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    banana_mask = cv2.morphologyEx(
        banana_mask, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))

    contours, hierarchy = cv2.findContours(
        banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= area:
            banana_counter += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_to_contours, (x, y),
                          (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image_to_contours, "BANANA", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return banana_counter, banana_mask


def FindApple(filtered_image, fruits_detected, area, image_to_contours):
    # Cutting out apples by specific thresholds
    apple_counter = 0

    apple_l = (0, 79, 17)
    apple_h = (18, 209, 220)
    apple_mask1 = cv2.inRange(filtered_image, apple_l, apple_h)
    apple_mask1 = cv2.morphologyEx(
        apple_mask1, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    apple_mask1 = cv2.morphologyEx(
        apple_mask1, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))
    apple_mask1 = cv2.bitwise_and(apple_mask1, fruits_detected)

    apple_l2 = (80, 76, 0)
    apple_h2 = (179, 255, 255)
    apple_mask2 = cv2.inRange(filtered_image, apple_l, apple_h)
    apple_mask2 = cv2.morphologyEx(
        apple_mask2, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    apple_mask2 = cv2.morphologyEx(
        apple_mask2, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))

    apple_mask = cv2.bitwise_or(apple_mask1, apple_mask2)

    contours, hierarchy = cv2.findContours(
        apple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > area:
            apple_counter += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_to_contours, (x, y),
                          (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_to_contours, "APPLE", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return apple_counter, apple_mask


def FindOrange(filtered_image, fruits_detected, area, image_to_contours):
    # Cutting out oranges by specific thresholds
    orange_counter = 0
    orange_l = (11, 65, 152)
    orange_h = (19, 255, 255)
    orange_mask = cv2.inRange(filtered_image, orange_l, orange_h)
    orange_mask = cv2.morphologyEx(
        orange_mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    orange_mask = cv2.morphologyEx(
        orange_mask, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))
    orange_mask = cv2.bitwise_and(orange_mask, fruits_detected)

    contours, hierarchy = cv2.findContours(
        orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > area:
            orange_counter += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_to_contours, (x, y),
                          (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image_to_contours, "ORANGE", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return orange_counter, orange_mask


def detect_fruits(img_path: str):
    # Main function which is cutting out and count fruits from image
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """

    image_r = ReadyImage(img_path, cv2.INTER_CUBIC, 2)
    image_fil = ImageFiltering(image_r)
    fruits_detected = FruitsDetection(image_fil)

    area = image_r.shape[1]*image_r.shape[0]*0.005
    banana, ban_mask = FindBanana(image_fil, fruits_detected, area, image_r)
    fruits_detected = fruits_detected-ban_mask
    apple, app_mask = FindApple(image_fil, fruits_detected, area, image_r)
    fruits_detected = fruits_detected - app_mask
    orange, oran_mask = FindOrange(image_fil, fruits_detected, area, image_r)

    return {'apple': apple, 'banana': banana, 'orange': orange}


def main():
    # Creating list of images from folder 'data'
    im_list = os.listdir('./data')
    im_list.sort()

    result = {}
    for image in im_list:
        full_path = os.path.join('data/', image)
        result[image] = (detect_fruits(full_path))

    with open('results.json', 'w') as file:
        json.dump(result, file)

    print(result, sep='\n')


if __name__ == '__main__':
    main()
