import numpy as np
import cv2
import torch


def randomcrop(image, mask, min_ratio=0.5):
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
    if mask is not None:
        mask = mask[y1:y2, x1:x2]
    return image, mask

def coarse_dropout(image, mask=None, num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), value=0):

    image = image.copy()
    if mask is not None:
        mask = mask.copy()
    num_holes = np.random.randint(num_holes_range[0], num_holes_range[1]+1)
    h, w, c = image.shape[-3:]

    for _ in range(num_holes):

        hole_h = np.random.randint(h*hole_height_range[0], h*hole_height_range[1]+1)
        hole_w = np.random.randint(w*hole_width_range[0], w*hole_width_range[1]+1)

        y = np.random.randint(0, h - hole_h+1)
        x = np.random.randint(0, w - hole_w+1)

        # 구멍을 검은색(0)으로 채우기
        image[..., y:y + hole_h, x:x + hole_w, :] = value
        if mask is not None:
            mask[..., y:y + hole_h, x:x + hole_w, :] = 255

    return image, mask

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    s_factor = 1.0 + np.random.uniform(-saturation, saturation)
    h_factor = np.random.uniform(-hue * 180, hue * 180)

    # Brightness
    image = np.clip(image * b_factor, 0, 255).astype(np.uint8)

    # Contrast
    mean = np.mean(image, axis=(0, 1), keepdims=True)
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

def rotate(image, mask=None, degree=0, value=(255, 255, 255)):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    rot_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(image, rot_matrix, (new_w, new_h), borderValue=value)

    rotated_mask = None
    if mask is not None:
        if len(mask.shape) == 2:
            border_val = 255
        else:
            border_val = (255,) * mask.shape[2]

        rotated_mask = cv2.warpAffine(mask, rot_matrix, (new_w, new_h), borderValue=border_val)

    return rotated_image, rotated_mask

def flip(image):
    return image[:, ::-1]

def identity(image):
    return image

def gaussian_noise(image, mean=0.0, std=1.0):
    noise = np.random.normal(loc=mean, scale=std, size=image.shape)
    noisy_data = np.clip(image + noise, 0, 255)
    return noisy_data
    
def rgb_shift(image, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20):

    image = image.astype(np.int16) 

    r_shift = np.random.randint(-r_shift_limit, r_shift_limit + 1)
    g_shift = np.random.randint(-g_shift_limit, g_shift_limit + 1)
    b_shift = np.random.randint(-b_shift_limit, b_shift_limit + 1)

    image[..., 0] = np.clip(image[..., 0] + b_shift, 0, 255)  # B
    image[..., 1] = np.clip(image[..., 1] + g_shift, 0, 255)  # G
    image[..., 2] = np.clip(image[..., 2] + r_shift, 0, 255)  # R

    return image.astype(np.uint8)

def scale(image, scale=1.0, value=0):

    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        cropped = resized[start_y:start_y + h, start_x:start_x + w]
        return cropped

    elif scale < 1.0:
        canvas = np.zeros_like(image)+value 
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        return canvas

    else:
        return image  

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, shuffled_y
    return x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):  
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam