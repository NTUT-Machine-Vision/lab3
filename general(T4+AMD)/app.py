import gradio as gr
from functools import partial
import getpass
import os
import pandas as pd
import socket

from paper_store import PaperStore
from handlers import (
    handle_pdf_processing,
    handle_load_initial_search_view,
    handle_search_query,
    handle_display_paper_details,
    handle_model_test,
)
from constants import MODELS, SEL_PAPER_MSG, CUSTOM_CSS, LABEL_MAP, MODELS_INSTANCES
from api_model import APIModel

# Initialize the paper store (a single instance for the app's lifecycle)
paper_store_instance = PaperStore()

# Use functools.partial to pre-fill the paper_store argument for our handlers.
_handle_pdf_processing = partial(
    handle_pdf_processing, paper_store=paper_store_instance
)
_handle_load_initial_search_view = partial(
    handle_load_initial_search_view, paper_store=paper_store_instance
)
_handle_search_query = partial(handle_search_query, paper_store=paper_store_instance)
_handle_display_paper_details = partial(
    handle_display_paper_details, paper_store=paper_store_instance
)


def test_port(host, port) -> bool:
    """test if a port is open on a given host"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            if result == 0:
                print(f"Port {port} is open on {host}")
            else:
                print(f"Port {port} is closed on {host}")
            return result == 0
    except Exception as e:
        print(f"Error checking port {port} on {host}: {e}")
        return False


# Remove unavailable models
remove_models = []
for model in MODELS:
    model_instance = MODELS_INSTANCES.get(model)
    if isinstance(model_instance, APIModel):
        print(f"Checking availability of model: {model}")
        if not test_port(model_instance.host, model_instance.port):
            print(f"Model {model} is unavailable, removing it.")
            remove_models.append(model)
for model in remove_models:
    print(f"Removing unavailable model: {model}")
    MODELS_INSTANCES.pop(model)
    MODELS.remove(model)

with gr.Blocks(theme=gr.themes.Default(), css=CUSTOM_CSS) as app:
    gr.Markdown("# PDF Document Analyzer")
    with gr.Tab("Test model") as model_test_tab:

        data = {
            "項目": ["推理時間"] + [f"{k} count" for k in LABEL_MAP.keys()],
        }

        for name in MODELS:
            data[name] = [0] * (len(LABEL_MAP) + 1)

        pdf = gr.File(label="Upload PDF", file_types=[".pdf"])

        btn = gr.Button("Run Model", variant="primary")

        table = gr.DataFrame(pd.DataFrame(data), interactive=False)

        gallary = gr.Gallery(
            show_label=False,
            columns=3,
            object_fit="contain",
            height="auto",
        )

        btn.click(
            fn=handle_model_test,
            inputs=[pdf, table],
            outputs=[table, gallary],
        )

    with gr.Tab("Upload and Analyze") as upload_tab:
        model_choice = gr.Radio(label="Select Model", choices=MODELS, value=MODELS[0])
        pdf_upload = gr.File(
            label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple"
        )
        run_button = gr.Button("Run Analysis", variant="primary")
        status_log = gr.Textbox(label="Status Log", lines=8, interactive=False)
        run_button.click(
            fn=_handle_pdf_processing,
            inputs=[model_choice, pdf_upload],
            outputs=[status_log, pdf_upload],
            show_progress="full",
        )
    with gr.Tab("Search and View") as search_tab:
        with gr.Column():
            with gr.Row(elem_id="search-bar-container"):
                search_input = gr.Textbox(
                    placeholder="Type keyword to search papers... (leave empty to list all papers)",
                    show_label=False,
                    container=False,
                )
                search_button = gr.Button(
                    "Search", variant="primary", elem_id="search-button"
                )
            paper_radio = gr.Radio(
                label="Relevant Papers",
                choices=[],
                value=None,
                interactive=True,
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    paper_desc = gr.Textbox(
                        lines=20,
                        container=False,
                        interactive=False,
                        placeholder=SEL_PAPER_MSG,
                        elem_id="description-textbox",
                    )
                with gr.Column(scale=2, min_width=400):
                    image_gallery = gr.Gallery(
                        show_label=False,
                        columns=3,
                        object_fit="contain",
                        height="auto",
                        elem_id="image-gallery-component",
                    )

        search_outputs = [paper_radio, paper_desc, image_gallery]
        search_button.click(
            fn=_handle_search_query, inputs=[search_input], outputs=search_outputs
        )
        search_input.submit(
            fn=_handle_search_query, inputs=[search_input], outputs=search_outputs
        )
        paper_radio.change(
            fn=_handle_display_paper_details,
            inputs=[paper_radio],
            outputs=[paper_desc, image_gallery],
            show_progress="full",
        )
        search_tab.select(
            fn=_handle_load_initial_search_view,
            inputs=None,
            outputs=search_outputs,
            queue=True,
        )

    app.load(fn=_handle_load_initial_search_view, inputs=None, outputs=search_outputs)


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter your Gemini API key (will not be stored): "
        )
    app.launch(debug=True)
