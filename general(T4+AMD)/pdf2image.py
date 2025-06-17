import fitz
from PIL import Image
import numpy as np


def pdf_to_images(pdf_path: str, zoom_x=2, zoom_y=2) -> list[np.ndarray]:
    """Convert PDF pages to images with increased resolution."""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)
            images.append(img_array)
        doc.close()
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    return images
