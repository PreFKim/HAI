import numpy as np
import cv2


def randomcrop(image, min_ratio=0.5):
    h, w, c = image.shape

    cx = np.random.uniform()*(1-min_ratio) * w
    cy = np.random.uniform()*(1-min_ratio) * h

    cw = np.random.uniform(min_ratio, 1) * (w+1)
    ch = np.random.uniform(min_ratio, 1) * (h+1)

    x1 = int(cx)
    y1 = int(cy)

    x2 = int(cx+cw)
    y2 = int(cy+ch)

    image = image[y1:y2, x1:x2]
    return image

def coarse_dropout(image, num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), value=0): # inplace 연산임

    num_holes = np.random.randint(num_holes_range[0], num_holes_range[1]+1)
    h, w, c = image.shape[-3:]

    for _ in range(num_holes):

        hole_h = np.random.randint(h*hole_height_range[0], h*hole_height_range[1]+1)
        hole_w = np.random.randint(w*hole_width_range[0], w*hole_width_range[1]+1)

        y = np.random.randint(0, h - hole_h+1)
        x = np.random.randint(0, w - hole_w+1)

        # 구멍을 검은색(0)으로 채우기
        image[..., y:y + hole_h, x:x + hole_w, :] = value

    return image

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    s_factor = 1.0 + np.random.uniform(-saturation, saturation)
    h_factor = np.random.uniform(-hue * 180, hue * 180)

    # Brightness
    image = np.clip(image * b_factor, 0, 255).astype(np.uint8)

    # Contrast
    mean = np.mean(image, axis=(-3, -2), keepdims=True)
    image = np.clip((image - mean) * c_factor + mean, 0, 255).astype(np.uint8)

        
    # Saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_factor, 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Hue
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + h_factor) % 180
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return image

def rotate(image, degree):

    h, w, c = image.shape
    center = (w / 2, h / 2)  
    rot_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

    image = cv2.warpAffine(image, rot_matrix, (w, h))

    return image

def flip(image):
    return image[:, ::-1]

def identity(image):
    return image

def scale(image, scale=1.0) -> np.ndarray:

    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        # 중심 crop
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        cropped = resized[start_y:start_y + h, start_x:start_x + w]
        return cropped

    elif scale < 1.0:
        # 중심에 paste (padding)
        canvas = np.zeros_like(image)  # 검은 배경
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return canvas

    else:
        return image  # scale == 1.0
