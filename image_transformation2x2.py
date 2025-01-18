import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class ImageStitching:
    def __init__(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.img.shape

    def processing(self, new_h, new_w, matrix, min_x=0, min_y=0):
        output_img = np.zeros((new_h, new_w), dtype=np.uint8)
        for y in range(self.h):
            for x in range(self.w):
                vitribandau = np.array([x, y], dtype=np.float32)
                vitrimoi = matrix @ vitribandau
                new_x, new_y = vitrimoi.astype(int)
                output_img[new_y - min_y, new_x - min_x] = self.img[y, x]
        return output_img

    def scaling_image(self, a, b):
        new_h = int(self.h * b)
        new_w = int(self.w * a)
        matrix = np.array([
            [a, 0],
            [0, b]
        ], dtype=np.float32)
        return self.processing(new_h, new_w, matrix)

    def rotation_image(self, phi):
        new_h = int(math.sqrt(self.w**2 + self.h**2))
        new_w = int(math.sqrt(self.w**2 + self.h**2))
        matrix = np.array([
            [math.cos(phi), -math.sin(phi)],
            [math.sin(phi), math.cos(phi)],
        ], dtype=np.float32)
        check_min_x = []
        check_min_y = []
        for y in range(self.h):
            for x in range(self.w):
                vitribandau = np.array([x, y], dtype=np.float32)
                vitrimoi = matrix @ vitribandau
                new_x, new_y = vitrimoi.astype(int)
                if new_x < 0:
                    check_min_x.append(new_x)
                if new_y < 0:
                    check_min_y.append(new_y)
        min_x = min(check_min_x) if check_min_x else 0
        min_y = min(check_min_y) if check_min_y else 0
        return self.processing(new_h, new_w, matrix, min_x, min_y)

    def skew_image(self, axis, m):
        new_h = self.w * m + self.h
        new_w = self.h * m + self.w
        if axis == 1:  # Horizontal
            matrix = np.array([
                [1, m],
                [0, 1]
            ], dtype=np.float32)
        else:  # Vertical
            matrix = np.array([
                [1, 0],
                [m, 1]
            ], dtype=np.float32)
        return self.processing(int(new_h), int(new_w), matrix)

    def mirror_image(self, axis):
        new_h = self.h
        new_w = self.w
        if axis == 1:  # Axis y
            matrix = np.array([
                [-1, 0],
                [0, 1]
            ], dtype=np.float32)
        else:  # Axis Y=X
            matrix = np.array([
                [0, 1],
                [1, 0]
            ], dtype=np.float32)
        return self.processing(new_h, new_w, matrix)

if __name__ == '__main__':
    img = cv2.imread('cat.jpg',cv2.COLOR_BGR2GRAY)
    processor = ImageStitching(img)

    scaled_image = processor.scaling_image(2, 1)
    rotated_image = processor.rotation_image(math.radians(45))
    skewed_image_h = processor.skew_image(1, 0.5)
    skewed_image_v = processor.skew_image(0, 0.5)
    mirrored_image_y = processor.mirror_image(1)
    mirrored_image_diag = processor.mirror_image(0)

    images = [
        ("Original Image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ("Scaled Image", scaled_image),
        ("Rotated Image (45°)", rotated_image),
        ("Skewed Image (Horizontal)", skewed_image_h),
        ("Skewed Image (Vertical)", skewed_image_v),
        ("Mirrored Image (Y-axis)", mirrored_image_y),
        ("Mirrored Image (Diagonal)", mirrored_image_diag),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs = axs.ravel()

    for i, (title, image) in enumerate(images):
        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(title)
        axs[i].axis('off')

    for j in range(len(images), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
