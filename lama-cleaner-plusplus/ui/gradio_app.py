import gradio as gr
import numpy as np
from PIL import Image

from config import AppConfig, load_config
from core.service import InpaintService
from core.mask_processor import MaskProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_mask_from_editor(editor_data: dict, image: Image.Image) -> np.ndarray:
    h, w = image.height, image.width

    layers = editor_data.get("layers", [])
    mask_acc = None

    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, dict):
            if "data" in layer:
                layer = layer["data"]
            else:
                continue
        if isinstance(layer, np.ndarray):
            if layer.ndim == 2:
                alpha = layer.astype(np.float32) / 255.0
            elif layer.ndim == 3 and layer.shape[2] >= 4:
                alpha = layer[:, :, 3].astype(np.float32) / 255.0
            elif layer.ndim == 3:
                alpha = np.mean(layer[:, :, :3], axis=2).astype(np.float32) / 255.0
            else:
                logger.warning(f"Skipping layer with unexpected ndim={layer.ndim}, shape={layer.shape}")
                continue
        elif hasattr(layer, "mode"):
            if layer.mode == "RGBA":
                alpha = np.array(layer)[:, :, 3].astype(np.float32) / 255.0
            elif layer.mode == "L":
                alpha = np.array(layer).astype(np.float32) / 255.0
            else:
                alpha = np.array(layer.convert("L")).astype(np.float32) / 255.0
        else:
            logger.warning(f"Skipping layer of unsupported type: {type(layer).__name__}")
            continue
        if mask_acc is None:
            mask_acc = alpha
        else:
            mask_acc = np.maximum(mask_acc, alpha)

    if mask_acc is None:
        composite = editor_data.get("composite")
        if composite is not None:
            if isinstance(composite, np.ndarray):
                if composite.ndim == 3 and composite.shape[2] >= 4:
                    alpha = composite[:, :, 3].astype(np.float32) / 255.0
                    if alpha.max() > 0:
                        comp_rgb = composite[:, :, :3].astype(np.float32)
                        bg_rgb = np.array(image).astype(np.float32)
                        diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                        threshold = max(8, diff.mean() * 0.5)
                        mask_acc = ((diff > threshold) & (alpha > 0)).astype(np.float32)
                    else:
                        mask_acc = alpha
                elif composite.ndim == 3:
                    comp_rgb = composite[:, :, :3].astype(np.float32)
                    bg_rgb = np.array(image).astype(np.float32)
                    diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                    threshold = max(8, diff.mean() * 0.5)
                    mask_acc = (diff > threshold).astype(np.float32)
            elif hasattr(composite, "mode"):
                if composite.mode == "RGBA":
                    alpha = np.array(composite)[:, :, 3].astype(np.float32) / 255.0
                    if alpha.max() > 0:
                        comp_rgb = np.array(composite)[:, :, :3].astype(np.float32)
                        bg_rgb = np.array(image).astype(np.float32)
                        diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                        threshold = max(8, diff.mean() * 0.5)
                        mask_acc = ((diff > threshold) & (alpha > 0)).astype(np.float32)
                    else:
                        mask_acc = alpha
                elif composite.mode == "RGB":
                    comp_rgb = np.array(composite).astype(np.float32)
                    bg_rgb = np.array(image).astype(np.float32)
                    diff = np.abs(comp_rgb - bg_rgb).max(axis=2)
                    threshold = max(8, diff.mean() * 0.5)
                    mask_acc = (diff > threshold).astype(np.float32)

    if mask_acc is None or mask_acc.max() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask_np = (mask_acc * 255).astype(np.uint8)
    mask_np = np.where(mask_np > 128, 255, 0).astype(np.uint8)
    return mask_np


def create_app(config: AppConfig | None = None) -> gr.Blocks:
    config = config or load_config()
    service = InpaintService(config)

    with gr.Blocks(title="Lama Cleaner++") as app:
        gr.Markdown("# 🖼️ Lama Cleaner++")
        gr.Markdown("AI 局部图像修复工具 — 圈哪里修哪里")

        with gr.Row():
            with gr.Column(scale=1):
                input_editor = gr.ImageEditor(
                    label="上传图片 & 画 Mask",
                    type="pil",
                    image_mode="RGB",
                    brush=gr.Brush(
                        default_size=20,
                        colors=["#FFFFFF"],
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=20),
                    layers=True,
                    sources=["upload", "clipboard"],
                    height=500,
                )
                prompt = gr.Textbox(
                    label="Prompt（可选，有默认值）",
                    placeholder="描述修复后的效果，留空使用默认值",
                    value="",
                )
                with gr.Row():
                    mode = gr.Radio(
                        ["auto", "remove", "quick", "hq", "cpu"],
                        value="auto",
                        label="模式",
                    )
                    seed = gr.Number(
                        label="Seed（默认 42）",
                        value=42,
                        precision=0,
                    )
                gr.Markdown(
                    "**模式说明：** auto=自动选择 | remove=纯去除(LaMa) | quick=快速生成(SDXL) | hq=高质量生成(SDXL) | cpu=纯CPU"
                )
                with gr.Accordion("高级参数", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt（可选）",
                        placeholder="描述需要避免的效果，留空使用默认值",
                        value="",
                    )
                    expand_slider = gr.Slider(
                        0, 50, value=config.mask.expand_pixels,
                        step=1, label="Mask 膨胀 (px)",
                    )
                    feather_slider = gr.Slider(
                        0, 50, value=config.mask.feather_radius,
                        step=1, label="Mask 羽化 (px)",
                    )
                    upscale = gr.Radio(
                        ["none", "output", "input", "only"],
                        value="none",
                        label="超分辨率 (RealESRGAN x4)",
                    )
                    gr.Markdown(
                        "**超分说明：** none=不超分 | output=修复结果超分 | input=先超分输入再修复 | only=仅超分不修复"
                    )
                with gr.Row():
                    run_btn = gr.Button("🚀 Run", variant="primary", size="lg")
                    undo_btn = gr.Button("↩️ Undo", size="sm")

            with gr.Column(scale=1):
                output_image = gr.Image(label="修复结果", type="pil")
                mask_preview = gr.Image(label="Mask 预览", type="pil")

        history_state = gr.State([])

        status_text = gr.Textbox(
            label="状态",
            value="就绪",
            interactive=False,
            max_lines=1,
        )

        _progress_map = [
            ("分析 mask", 0.15),
            ("加载", 0.35),
            ("推理", 0.65),
            ("LaMa", 0.65),
            ("拼接", 0.85),
            ("完成", 1.0),
        ]

        def on_run(editor_data, prompt_text, neg_prompt_text, mode_val, seed_val, expand_val, feather_val, upscale_val, hist, progress=gr.Progress()):
            if editor_data is None:
                return "❌ 无输入图片", None, None, hist
            if isinstance(editor_data, Image.Image):
                return "⚠️ 未检测到 mask，请先画 mask", None, None, hist
            elif isinstance(editor_data, dict):
                image = editor_data.get("background")
                if image is None:
                    return "❌ 无背景图片", None, None, hist
                mask_np = _extract_mask_from_editor(editor_data, image)
            else:
                return "❌ 不支持的输入格式", None, None, hist

            if mask_np.max() == 0:
                return "⚠️ 未检测到 mask，请先画 mask", None, None, hist

            mask = Image.fromarray(mask_np)
            progress(0.05, desc="⏳ 正在处理...")

            def on_status(msg):
                for keyword, value in _progress_map:
                    if keyword in msg:
                        progress(value, desc=msg)
                        break

            try:
                result, preview, hist = service.process(
                    image=image,
                    mask=mask,
                    prompt=prompt_text,
                    negative_prompt=neg_prompt_text,
                    mode=mode_val,
                    expand_px=int(expand_val),
                    feather_radius=int(feather_val),
                    seed=int(seed_val) if seed_val is not None else None,
                    history=hist,
                    status_callback=on_status,
                    upscale=upscale_val,
                )
                return "✅ 完成", result, preview, hist
            except RuntimeError as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                return f"❌ 处理失败: {e}", None, None, hist
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return f"❌ 意外错误: {e}", None, None, hist

        def on_undo(hist):
            result, hist = service.undo(hist)
            return result, None, hist

        run_btn.click(
            fn=on_run,
            inputs=[input_editor, prompt, negative_prompt, mode, seed, expand_slider, feather_slider, upscale, history_state],
            outputs=[status_text, output_image, mask_preview, history_state],
        )
        undo_btn.click(fn=on_undo, inputs=[history_state], outputs=[output_image, mask_preview, history_state])

    return app
