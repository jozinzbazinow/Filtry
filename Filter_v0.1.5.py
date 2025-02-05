import sys
import numpy as np
import cv2
from numba import njit, prange
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtCore import Qt

@njit
def reflect_pad(image, padding):
    """Reflect padding for the image."""
    H, W, C = image.shape
    padded_image = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=image.dtype)
    
    # Fill the center with the original image
    padded_image[padding:padding+H, padding:padding+W] = image
    
    # Reflect padding on the top and bottom
    for i in range(padding):
        padded_image[padding - 1 - i, :] = padded_image[padding + i, :]  # Top
        padded_image[padding + H + i, :] = padded_image[padding + H - 1 - i, :]  # Bottom
    
    # Reflect padding on the left and right
    for j in range(padding):
        padded_image[:, padding - 1 - j] = padded_image[:, padding + j]  # Left
        padded_image[:, padding + W + j] = padded_image[:, padding + W - 1 - j]  # Right
    
    return padded_image

@njit
def rgb_to_grayscale(rgb):
    """Convert RGB image to grayscale."""
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

@njit
def generate_8_slices(kernel_size):
    """Generate 8 triangular slices for the generalized Kuwahara filter."""
    center = kernel_size // 2
    slices = np.zeros((8, kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            y = i - center
            x = j - center
            angle = np.arctan2(y, x)  # Angle in radians (-pi to pi)
            angle = (angle + np.pi) / (2 * np.pi) * 8  # Map to [0, 8)
            slice_index = int(angle) % 8
            slices[slice_index, i, j] = 1.0
    return slices

@njit
def gaussian_kernel_2d(kernel_size, sigma):
    """Generate a 2D Gaussian kernel."""
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            y = i - center
            x = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize
    return kernel

@njit
def compute_gradient(image):
    """Compute the gradient magnitude and angle using Sobel operators."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    H, W = image.shape
    gradient_x = np.zeros((H, W), dtype=np.float32)
    gradient_y = np.zeros((H, W), dtype=np.float32)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            gradient_x[y, x] = np.sum(image[y-1:y+2, x-1:x+2] * sobel_x)
            gradient_y[y, x] = np.sum(image[y-1:y+2, x-1:x+2] * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angle = np.arctan2(gradient_y, gradient_x)  # Angle in radians
    return gradient_magnitude, gradient_angle

@njit(parallel=True)
def generalized_kuwahara_filter(image, kernel_size=5, sigma=1.0, q=1.0):
    """Apply the Generalized Kuwahara Filter."""
    H, W, C = image.shape
    radius = kernel_size // 2
    output = np.zeros((H, W, C), dtype=np.uint8)

    # Generate 8 slices and Gaussian kernel
    slices = generate_8_slices(kernel_size)
    gauss_kernel = gaussian_kernel_2d(kernel_size, sigma)

    # Convert image to grayscale for gradient computation
    grayscale = rgb_to_grayscale(image)

    # Compute gradient magnitude and angle
    gradient_magnitude, gradient_angle = compute_gradient(grayscale)

    # Pad the image to handle borders
    padded_image = reflect_pad(image, radius)

    for y in prange(radius, H + radius):
        for x in prange(radius, W + radius):
            for c in range(C):
                window = padded_image[y-radius:y+radius+1, x-radius:x+radius+1, c]
                local_angle = gradient_angle[y-radius, x-radius]

                # Determine which slice the local angle belongs to
                slice_index = int((local_angle + np.pi) / (2 * np.pi) * 8) % 8

                # Extract the region corresponding to the slice
                region = window * slices[slice_index]

                # Compute the weighted mean of the region
                weights = gauss_kernel * slices[slice_index]
                weighted_mean = np.sum(region * weights) / (np.sum(weights) + 1e-8)

                # Clip the value to the range [0, 255]
                output[y-radius, x-radius, c] = min(255, max(0, int(weighted_mean)))

    # Adjust brightness to match the input
    input_brightness = np.mean(image)
    output_brightness = np.mean(output)
    brightness_ratio = input_brightness / (output_brightness + 1e-8)
    output = np.clip(output * brightness_ratio, 0, 255).astype(np.uint8)
    
    return output

@njit
def bilateral_filter(image, kernel_size, sigma_space, sigma_color):
    H, W, C = image.shape
    padding = kernel_size // 2
    output = np.zeros_like(image, dtype=np.uint8)

    ax = np.linspace(-padding, padding, kernel_size)
    gauss_space = np.exp(-0.5 * (ax**2) / (sigma_space**2))
    gauss_space = np.outer(gauss_space, gauss_space)

    for y in range(padding, H - padding):
        for x in range(padding, W - padding):
            for c in range(C):
                window = image[y-padding:y+padding+1, x-padding:x+padding+1, c]

                intensity_diff = window - image[y, x, c]
                gauss_color = np.exp(-0.5 * (intensity_diff**2) / (sigma_color**2))

                weights = gauss_space * gauss_color
                weights /= np.sum(weights)

                output[y, x, c] = np.sum(window * weights)
    return output

@njit
def kuwahara_filter(image, kernel_size):
    H, W, C = image.shape
    padding = kernel_size // 2
    output = np.zeros_like(image)

    for y in range(padding, H - padding):
        for x in range(padding, W - padding):
            for c in range(C):
                window = image[y-padding:y+padding+1, x-padding:x+padding+1, c]

                mid = kernel_size // 2
                q1 = window[:mid+1, :mid+1]  
                q2 = window[:mid+1, mid:]    
                q3 = window[mid:, :mid+1]    
                q4 = window[mid:, mid:]      

                variances = np.array([np.var(q1), np.var(q2), np.var(q3), np.var(q4)])
                min_index = np.argmin(variances)

                quarters = [q1, q2, q3, q4]
                output[y, x, c] = np.mean(quarters[min_index])
    return output


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.original_image = None  
        self.processed_image = None
        self.kernel_size = 5
        self.sigma = 10.0
        self.current_filter = 0  # 0: Oryginalny, 1: Blur, 2: Gaussian, 3: Kuwahara 4:Generalized Kuwahara 5: Bilateral 
        self.filters = ["Oryginalny", "Blur", "Gaussian", "Kuwahara", "Generalized Kuwahara", "Bilateral"]

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Filtry obrazu")
        self.setAcceptDrops(True)

        self.filter_label = QLabel(self.filters[self.current_filter])
        self.filter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filter_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.label = QLabel("Przeciągnij i upuść zdjęcie tutaj")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.left_button = QPushButton("⬅")
        self.right_button = QPushButton("➡")

        self.left_button.clicked.connect(self.reset_to_original)
        self.right_button.clicked.connect(self.change_filter)

        self.slider_kernel = self.create_slider(3, 50, self.kernel_size, self.update_kernel)
        self.slider_sigma = self.create_slider(1, 50, int(self.sigma), self.update_sigma)

        self.label_kernel = QLabel(f"Kernel: {self.kernel_size}")
        self.label_sigma = QLabel(f"Sigma: {self.sigma}")

        top_layout = QVBoxLayout()
        top_layout.addWidget(self.filter_label)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_button)
        image_layout.addWidget(self.label)
        image_layout.addWidget(self.right_button)

        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(self.label_kernel)
        sliders_layout.addWidget(self.slider_kernel)
        sliders_layout.addWidget(self.label_sigma)
        sliders_layout.addWidget(self.slider_sigma)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(sliders_layout)

        self.setLayout(main_layout)
        self.resize(700, 500)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                self.load_image(file_path)
                break

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_image = image
        self.processed_image = image.copy()
        self.display_image(self.original_image)

    def create_slider(self, min_val, max_val, default, callback):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.valueChanged.connect(callback)
        return slider

    def update_kernel(self, value):
        if value % 2 == 0:
            adjusted_value = value + 1
        else:
            adjusted_value = value
        self.kernel_size = adjusted_value
        self.label_kernel.setText(f"Kernel: {value}")
        self.apply_current_filter()

    def update_sigma(self, value):
        self.sigma = value
        self.label_sigma.setText(f"Sigma: {value}")
        self.apply_current_filter()

    def change_filter(self):
        self.current_filter = (self.current_filter + 1) % len(self.filters)
        filter_name = self.filters[self.current_filter]

        if self.current_filter == 3 and self.kernel_size % 2 == 0:
            forced_kernel = self.kernel_size + 1
            self.filter_label.setText(f"{filter_name} (kernel forced to {forced_kernel} from {self.kernel_size})")
        else:
            self.filter_label.setText(filter_name)
        
        self.apply_current_filter()

    def apply_current_filter(self):
        if self.original_image is None:
            return
        
        image = self.original_image.copy()
        kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1

        if self.current_filter == 1:
            image = self.blur_filter(image, kernel_size)
            
        elif self.current_filter == 2:
            image = self.gaussian_blur_filter(image, kernel_size, self.sigma)
            
        elif self.current_filter == 3:
            image = kuwahara_filter(image, kernel_size=kernel_size)
            
        elif self.current_filter == 4:
            image = generalized_kuwahara_filter(image, kernel_size=kernel_size, sigma=self.sigma, q = 1.0)
            
        elif self.current_filter == 5:
            image = bilateral_filter(image, kernel_size=kernel_size, sigma_space = self.sigma, sigma_color = self.sigma)
            

        self.processed_image = image
        self.display_image(self.processed_image)
        
        output_path = f"filtered_{self.current_filter}.png"
        cv2.imwrite(output_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
        return output_path

    def reset_to_original(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.current_filter = 0
            self.filter_label.setText(self.filters[self.current_filter])
            self.display_image(self.original_image)

    def blur_filter(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        return cv2.filter2D(image, -1, kernel)

    def gaussian_blur_filter(self, image, kernel_size, sigma):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def display_image(self, image):
        height, width, channels = image.shape
        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
