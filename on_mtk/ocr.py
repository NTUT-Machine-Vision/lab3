import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

def ocr_image(image : np.ndarray) -> str:
    """
    Perform OCR on the given image and return the extracted text.
    
    :param image_path: Path to the image file.
    :return: Extracted text from the image.
    """
    try:
        results = reader.readtext(image, detail=1, paragraph=True)
        extracted_text = " ".join([result[1] for result in results])
        return extracted_text
    except Exception as e:
        return f"Error during OCR processing: {str(e)}"