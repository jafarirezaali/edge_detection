# Edge Detection 
# Author: Alireza Jafari (GitHub: https://github.com/jafarirezaali)
# Date: 2025

import numpy as np
import cv2
import matplotlib.pyplot as plt


class EdgeDetection:
    def gaussian_kernel(self, size, sigma=1):
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.sum()

    def apply_blur(self, image, kernel):
        return cv2.filter2D(image, -1, kernel)

    def make_blurred(self, image_address, kernel_size=5, sigma=1):
        img = cv2.imread(image_address)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = self.gaussian_kernel(kernel_size, sigma)
        blurred = self.apply_blur(gray, kernel)
        self._show_images([gray, blurred], ["Original", "Manual Gaussian Blur"])
        return blurred

    def convolve(self, img, kernel):
        return cv2.filter2D(img, cv2.CV_32F, kernel)

    def sobel(self, img):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        Gx = self.convolve(img, sobel_x)
        Gy = self.convolve(img, sobel_y)
        G = np.hypot(Gx, Gy)
        G = np.clip(G, 0, 255).astype(np.uint8)
        self._show_images([np.abs(Gx), np.abs(Gy), G], ["Gx", "Gy", "Gradient Magnitude"])
        return Gx, Gy, G

    def sobel_with_gradent(self, Gx, Gy):
        theta_deg = (np.rad2deg(np.arctan2(Gy, Gx)) + 180) % 180
        self._show_image(theta_deg, "Gradient Direction (degrees)", cmap='hsv', colorbar=True)
        return theta_deg

    def quantize_angles(self, theta_deg):
        direction = np.zeros_like(theta_deg, dtype=np.uint8)
        bins = [(0, 22.5, 0), (22.5, 67.5, 45), (67.5, 112.5, 90),
                (112.5, 157.5, 135), (157.5, 180, 0)]
        for low, high, val in bins:
            direction[(theta_deg >= low) & (theta_deg < high)] = val
        self._show_image(direction, "Quantized Gradient Directions", cmap='Set1', colorbar=True)
        return direction

    def non_maximum_suppression(self, G, directions):
        nms = np.zeros_like(G, dtype=np.uint8)
        offsets = {0: ((0, -1), (0, 1)),
                   45: ((-1, 1), (1, -1)),
                   90: ((-1, 0), (1, 0)),
                   135: ((-1, -1), (1, 1))}
        for angle, (b_off, a_off) in offsets.items():
            mask = directions == angle
            i, j = np.where(mask)
            for y, x in zip(i, j):
                if 1 <= y < G.shape[0] - 1 and 1 <= x < G.shape[1] - 1:
                    before = G[y + b_off[0], x + b_off[1]]
                    after = G[y + a_off[0], x + a_off[1]]
                    if G[y, x] >= before and G[y, x] >= after:
                        nms[y, x] = G[y, x]
        self._show_image(nms, "Non-Maximum Suppression Result")
        return nms

    def double_threshold(self, img, low_thresh, high_thresh):
        strong, weak = 255, 75
        strong_mask = img >= high_thresh
        weak_mask = (img >= low_thresh) & ~strong_mask
        result = np.zeros_like(img, dtype=np.uint8)
        result[strong_mask] = strong
        result[weak_mask] = weak
        return result, weak, strong

    def hysteresis(self, img, weak, strong):
        h, w = img.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if img[y, x] == weak:
                    if np.any(img[y-1:y+2, x-1:x+2] == strong):
                        img[y, x] = strong
                    else:
                        img[y, x] = 0
        return img

    def canny(self, img, low=30, high=100):
        dt_result, weak, strong = self.double_threshold(img, low, high)
        final_edges = self.hysteresis(dt_result.copy(), weak, strong)
        self._show_image(final_edges, "Final Edges (Canny-style)")
        return final_edges

    def _show_images(self, imgs, titles, cmap='gray'):
        plt.figure(figsize=(5 * len(imgs), 4))
        for idx, (img, title) in enumerate(zip(imgs, titles)):
            plt.subplot(1, len(imgs), idx+1)
            plt.imshow(img, cmap=cmap)
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _show_image(self, img, title, cmap='gray', colorbar=False):
        plt.figure(figsize=(6, 5))
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        if colorbar:
            plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
