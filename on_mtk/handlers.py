import gradio as gr
import time
import os
from typing import List, Tuple, Optional
import numpy as np
import pypdf
import cv2

from paper_store import PaperStore
from constants import PROMPT, MODELS_INSTANCES, NO_RES_ID, NO_RES_MSG, SEL_PAPER_MSG, LABEL_MAP
import pdf2image
import ocr
import gemini


def _create_ui_updates(
    choices: List[Tuple[str, str]],
    sel_id: Optional[str],
    desc_text: str,
    imgs: Optional[List[str]] = None,
):
    """Helper to create a tuple of Gradio update objects for search/display."""
    if imgs is None:
        imgs = []
    return gr.update(choices=choices, value=sel_id), desc_text, imgs


def handle_pdf_processing(
    model: str, pdf_files: Optional[list], paper_store: PaperStore
):
    """Processes uploaded PDF files, updates the paper store, and returns logs."""
    if not pdf_files:
        return "No PDFs provided for processing.", gr.update(value=None)

    logs = []
    temp_files_to_clean = []

    for pdf_obj in pdf_files:
        temp_files_to_clean.append(pdf_obj.name)
        filename = getattr(pdf_obj, "orig_name", os.path.basename(pdf_obj.name))
        existing_id = paper_store.find_paper_id_by_title(filename)

        if existing_id:
            logs.append(
                f"Paper '{filename}' already exists with ID: {existing_id}. Skipping reprocessing."
            )
        else:

            # process the PDF file
            images = pdf2image.pdf_to_images(pdf_obj.name)

            # use YOLO or RTDETR model for object detection
            model_instance = MODELS_INSTANCES[model]
            figures = []
            titles = []
            chapters = []
            for img in images:
                results = model_instance(img)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cropped_image = img[y1:y2, x1:x2]
                        if (
                            model_instance.names[class_id] == "fig"
                            or model_instance.names[class_id] == "table"
                        ):
                            figures.append(cropped_image)
                        elif model_instance.names[class_id] == "title":
                            titles.append(cropped_image)
                        elif model_instance.names[class_id] == "chapter":
                            chapters.append(cropped_image)

            # Perform OCR on the title and chapter images
            title_texts = [ocr.ocr_image(title) for title in titles]
            chapter_texts = [ocr.ocr_image(chapter) for chapter in chapters]

            # Combine all texts for context
            reader = pypdf.PdfReader(pdf_obj.name)
            texts = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
            combined_text = " ".join(texts)
            generated_text = gemini.generate_text(
                PROMPT.replace("{{CONTEXT}}", combined_text)
                .replace("{{TITLE}}", " ".join(title_texts))
                .replace("{{CHAPTERS}}", " ".join(chapter_texts)),
            )

            # get keywords from the generated text
            generated_keywords = gemini.generate_text(
                "Extract keywords from the following text in a comma-separated format, please do not include any additional text or formatting:\n\n"
                f"text: {generated_text}",
            )
            # Clean up the keywords
            keywords = [
                kw.strip() for kw in generated_keywords.split(",") if kw.strip()
            ]
            logs.append(f"Extracted keywords: {', '.join(keywords)}")

            paper_id = paper_store.add_paper(
                filename,
                generated_text,
                figures,
                keywords,
            )
            logs.append(
                f"Processed '{filename}' with model '{model}'. Added to store with ID: {paper_id}."
            )

    # After processing, the file input should be cleared
    return "\n".join(logs), gr.update(value=None)


def handle_load_initial_search_view(paper_store: PaperStore):
    """Loads the initial state of the search view components."""
    choices = paper_store.get_paper_choices()

    if len(choices) == 1 and choices[0][1] == NO_RES_ID:  # Only "No results" entry
        return _create_ui_updates(choices, NO_RES_ID, NO_RES_MSG)
    else:  # Papers exist or store is empty but not yet searched
        return _create_ui_updates(choices, None, SEL_PAPER_MSG)


def handle_search_query(keyword: str, paper_store: PaperStore):
    """Handles paper search based on a keyword."""
    results = paper_store.search_papers(
        keyword
    )  # Store now handles empty keyword by returning all

    if len(results) == 1 and results[0][1] == NO_RES_ID:  # No results found
        return _create_ui_updates(results, NO_RES_ID, NO_RES_MSG)
    else:  # Results found or all papers listed (for empty keyword)
        return _create_ui_updates(results, None, SEL_PAPER_MSG)


def handle_display_paper_details(paper_id: Optional[str], paper_store: PaperStore):
    """Displays details (description and images) for the selected paper."""
    if not paper_id or paper_id == NO_RES_ID:
        desc_text = NO_RES_MSG if paper_id == NO_RES_ID else SEL_PAPER_MSG
        return desc_text, []

    paper_info = paper_store.get_paper_details_by_id(paper_id)
    if not paper_info:
        # This case should be rare if UI and store are in sync
        return f"Error: Could not find data for paper ID '{paper_id}'.", []

    return paper_info.get("desc", "No description available."), paper_info.get(
        "imgs", []
    )

def handle_model_test(pdf,data):
    # clear data
    data.loc[1:,data.columns[1:]] = 0

    if pdf is None:
        return data, []
    
    def count_classes(results,model_name):
        """Counts the number of detected classes."""
        class_counts = {}
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                row_index = LABEL_MAP[class_name]+1
                data.loc[row_index, model_name] += 1
        return class_counts

    def mix_images_grid(images, cols=3):
        """排列圖片為網格形式，每列 cols 張。"""
        if not images:
            return None

        # 確保所有圖片大小一致
        h, w, c = images[0].shape
        for i in range(len(images)):
            if images[i].shape != (h, w, c):
                images[i] = cv2.resize(images[i], (w, h))  # 強制 resize 一致

        # 計算需要幾列
        rows = (len(images) + cols - 1) // cols

        # 補空圖使總數可被 cols 整除
        blank = np.zeros((h, w, c), dtype=np.uint8)
        while len(images) < rows * cols:
            images.append(blank)

        # 每 row 拼橫圖
        row_imgs = [np.hstack(images[i * cols:(i + 1) * cols]) for i in range(rows)]

        # 拼成整個圖
        grid_image = np.vstack(row_imgs)
        return grid_image

    images = pdf2image.pdf_to_images(pdf.name)

    all_image = [[img] for img in images]

    for model_name, model_instance in MODELS_INSTANCES.items():
        # 計算推論時間
        t1 = time.time()
        if hasattr(model_instance, "interpreter"):  # 判斷是否是 tflite 類型
            # 單張推論版本
            results = []
            for image in images:
                results.extend(model_instance([image]))  # 單張包成 list
        else:
            results = model_instance(images)  # 批次推論

        t2 = time.time()
        # 產生圖片
        for i,result in enumerate(results):
            img_with_boxes = result.plot()
            # draw model name
            cv2.putText(img_with_boxes, model_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            all_image[i].append(img_with_boxes)
        count_classes(results,model_name)
        data.loc[0,model_name] = f"{t2-t1:.2f} seconds"

    return data,[mix_images_grid(images) for images in all_image]
