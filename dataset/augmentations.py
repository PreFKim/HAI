import numpy as np
import cv2

def pad(image, target_size=(192, 256)):
    *n, h, w, c = image.shape
    tw, th = target_size

    if h/w > 1:
        padded = np.zeros((*n, h, int(h/th*tw), 3))
        x1 = (padded.shape[-2] - w)//2
        padded[:, :, x1:x1+w, :] = image
    else:
        padded = np.zeros((*n, int(w/tw*th), w, 3))
        y1 = (padded.shape[-3] - h)//2
        padded[:, y1:y1+h, :, :] = image

    return padded

def randomcrop(image):
    h, w, c = image.shape
    return image


def coarse_dropout(image, num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2)): # inplace 연산임

    num_holes = np.random.randint(num_holes_range[0], num_holes_range[1]+1)
    h, w, c = image.shape[-3:]

    for _ in range(num_holes):

        hole_h = np.random.randint(h*hole_height_range[0], h*hole_height_range[1]+1)
        hole_w = np.random.randint(w*hole_width_range[0], w*hole_width_range[1]+1)

        y = np.random.randint(0, h - hole_h+1)
        x = np.random.randint(0, w - hole_w+1)

        # 구멍을 검은색(0)으로 채우기
        image[..., y:y + hole_h, x:x + hole_w, :] = 0

    return image

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    # Brightness
    if len(image.shape) != 4 :
        raise ValueError("The length of image shape should be over 3 (n, h, w, c)")
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    s_factor = 1.0 + np.random.uniform(-saturation, saturation)
    h_factor = np.random.uniform(-hue * 180, hue * 180)

    # Brightness
    image = np.clip(image * b_factor, 0, 255).astype(np.uint8)

    # Contrast
    mean = np.mean(image, axis=(-3, -2), keepdims=True)
    image = np.clip((image - mean) * c_factor + mean, 0, 255).astype(np.uint8)

        
    for i in range(len(image)):
        # Saturation
        hsv = cv2.cvtColor(image[i], cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * s_factor, 0, 255)
        image[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Hue
        hsv = cv2.cvtColor(image[i], cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + h_factor) % 180
        image[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return image

def rotate(image, degree):

    h, w = image.shape[-3:-1]
    center = (w / 2, h / 2)  
    rot_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

    for i in range(len(image)):
        image[i] = cv2.warpAffine(image[i], rot_matrix, (w, h))

    return image
