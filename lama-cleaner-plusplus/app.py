import gradio as gr

from config import load_config
from ui.gradio_app import create_app
from utils.logger import setup_logger


def main():
    config = load_config()
    setup_logger("lama-cleaner-plusplus", config.log_level)
    app = create_app(config)
    app.launch(
        server_port=config.server_port,
        share=config.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
