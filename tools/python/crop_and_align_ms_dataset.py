import cv2
import time
import random
import os
import math
import base64
import struct
import Image
import numpy as np

fid = open("/home/salt/Downloads/TSV_MsCelebV1-Faces-Aligned/MsCelebV1-Faces-Aligned.tsv", "r")
base_path = '/home/salt/extracted/ms_full'

if not os.path.exists(base_path):
    os.mkdir(base_path)

eyeCascade = cv2.CascadeClassifier("/home/salt/dev/clf_2d_gpu_detect/data/haar/haarcascade_eye.xml")
faceCascade = cv2.CascadeClassifier("/home/salt/dev/clf_2d_gpu_detect/data/haar/haarcascade_frontalface_alt.xml")
images_saved = 0
how_many_do_i_need = 2000


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


while images_saved <= how_many_do_i_need:
    line = fid.readline()
    if line:
        data_info = line.split('\t')
        filename = str(images_saved) + ".jpg"
        filename_cropped = str(images_saved) + "_cropped_" + ".jpg"
        img_data = data_info[6].decode("base64")
        output_file_path = base_path + "/" + filename
        output_file_path_cropped = base_path + "/" + filename_cropped

        output_path = os.path.dirname(output_file_path)
        output_path_cropped = os.path.dirname(output_file_path_cropped)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(output_path_cropped):
            os.mkdir(output_path_cropped)

        img_file = open(output_file_path, 'w')
        img_file.write(img_data)
        img_file.close()
        img = cv2.imread(output_file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10),
                                             flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        if len(faces) > 0 and len(faces) < 2:
            for (x, y, w, h) in faces:
                # cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                eyes = eyeCascade.detectMultiScale(roi_gray)
                height, width = roi_gray.shape[:2]

                if len(eyes) is 2 and width > 100:
                    eye_left_center_x = 0
                    eye_left_center_y = 0
                    eye_right_center_x = 0
                    eye_right_center_y = 0
                    count = 0
                    for (ex, ey, ew, eh) in eyes:
                        # cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        if count == 1:
                            eye_left_center_x = ex + ew / 2
                            eye_left_center_y = ey + eh / 2
                        else:
                            eye_right_center_x = ex + ew / 2
                            eye_right_center_y = ey + eh / 2
                        count += 1

                    cv2.imwrite(output_file_path_cropped, roi_gray)
                    image_to_be_scaled = Image.open(output_file_path_cropped)
                    CropFace(image_to_be_scaled, eye_left=(eye_left_center_x, eye_left_center_y),
                             eye_right=(eye_right_center_x, eye_right_center_y),
                             offset_pct=(0.25, 0.25),
                             dest_sz=(80, 80)).save(output_file_path_cropped)
                    img = cv2.imread(output_file_path_cropped, cv2.COLOR_RGB2GRAY)
                    BLACK = np.array([0, 0, 0], np.uint8)
                    dst = cv2.inRange(img, BLACK, BLACK)
                    mostly_black = cv2.countNonZero(dst)
                    if mostly_black > 50:
                        os.remove(output_file_path_cropped)
                    else:
                        print "Images generated %d" % images_saved
                        cv2.imshow('img', img)
                        images_saved += 1

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            os.remove(output_file_path)
    else:
        break

fid.close()
