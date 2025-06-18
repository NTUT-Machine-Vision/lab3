from ultralytics import YOLO, RTDETR

MODELS = ["YOLO-plain", "YOLO-large_batch", "YOLO-adam_optimizer","tf-plain", "tf-large_batch", "tf-adam_optimizer", "RTDETR"]
NO_RES_ID = "NO_RES_ID"
NO_RES_MSG = "No papers found matching the search query."
SEL_PAPER_MSG = "Please select a paper to view details."
MODELS_INSTANCES = {
    "YOLO-plain": YOLO("models/YOLO-plain.pt"),
    "YOLO-large_batch": YOLO("models/YOLO-large_batch.pt"),
    "YOLO-adam_optimizer": YOLO("models/YOLO-adam_optimizer.pt"),
    # "YOLO-plain_tflite": YOLO("models/best_float32.tflite"),
    # "YOLO-batch_tflite": YOLO("models/batch_float32.tflite"),
    # "YOLO-adam_tflite": YOLO("models/adamw_float32.tflite"),
    # "RTDETR": RTDETR("models/RTDETR.pt"),
}
CUSTOM_CSS = """
#search-bar-container > .gr-form {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: var(--spacing-sm);
}
#search-bar-container > .gr-form > div:first-child {
    flex-grow: 1;
    border: none !important;
    box-shadow: none !important;
}
#search-button {
    min-width: auto !important;
    padding-left: var(--spacing-lg) !important;
    padding-right: var(--spacing-lg) !important;
    flex-grow: 0;
    flex-shrink: 0;
}
#image-gallery-component:empty::before {
    text-align: center;
    color: var(--text-color-subdued);
    font-style: italic;
    padding-top: 20px;
}
#description-textbox { height: 400px; }
#description-textbox textarea {
    height: 100% !important;
    box-sizing: border-box;
    overflow-y: auto !important;
}
"""
PROMPT = """
You are an AI assistant specializing in academic literature analysis. Your task is to generate a concise, objective, and informative summary of an academic paper. This summary is intended for an academic audience.

**Input Provided:**
*   `Title`: The title of the paper.
*   `Chapters`: Excerpts from the paper's chapters. Note: These excerpts may be jumbled due to OCR processing and might not represent the full paper or its original sequence.
*   `Context`: Background information about the paper's field, problem domain, or significance.

**Instructions for Generating the Summary:**
1.  **Source Adherence**: Base the summary *strictly* on the provided `Title`, `Chapters`, and `Context`. Do not infer information not present or make assumptions beyond the provided text.
2.  **Core Content**: The summary must clearly articulate:
    *   The paper's main purpose or primary research question(s).
    *   The key methodologies or approaches employed.
    *   The most significant findings, results, or contributions (e.g., novel techniques, important discoveries, theoretical advancements).
3.  **Synthesis and Coherence**: Synthesize the information from all inputs into a coherent narrative. Despite the potentially jumbled nature of the `Chapters` input, the final summary must be logically structured and easy to follow.
4.  **Style and Tone**:
    *   **Objective and Neutral**: Present information factually. Avoid personal opinions, subjective interpretations, or evaluative language (e.g., "remarkable," "groundbreaking").
    *   **Clarity and Precision**: Use clear, precise language. If technical jargon is essential, ensure its meaning is clear from the context or briefly explain it if necessary for an academic audience outside the paper's immediate sub-field.
    *   **Conciseness**: Each sentence should contribute new, relevant information. Avoid redundancy.
5.  **Output Format**:
    *   A single, well-structured paragraph.
    *   Written in English.
    *   No bullet points, numbered lists, or other special formatting in the final summary.
6.  **Length**: The summary should be between 150 and 250 words, providing a comprehensive yet concise overview of the paper.

**Generate the summary based on the following:**
Title: {{TITLE}}
Chapters: {{CHAPTERS}}
Context of the paper: {{CONTEXT}}
"""

LABEL_MAP = {
     "title":0,
     "authors":1,
     "table":2,
     "fig":3,
     "formula":4,
     "chapter":5,
     "pages":6,
     "reference":7
}
