# Lama Cleaner++

基于 AI 的图像修复与超分辨率工具，支持去除水印、文字、多余物体，以及 4 倍超分辨率放大。

## 功能特性

- **图像修复** — 涂抹要去除的区域，AI 自动填充背景
- **超分辨率** — RealESRGAN x4 放大，支持仅超分或修复后超分
- **多引擎支持** — LaMa（纯去除）+ SDXL（生成式填充），自动选择最优引擎
- **显存优化** — 引擎互斥加载、大图分块处理，1GB 显存即可使用全部功能（去除+超分）
- **本地模型** — 所有模型从 ModelScope 下载到本地，无需 HuggingFace

## 引擎说明

| 引擎 | 用途 | 显存占用 | 速度 |
|------|------|---------|------|
| **LaMa** | 纯去除（水印、文字、小物体） | ~30 MB | 极快 |
| **SDXL** | 生成式填充（大面积补全） | ~5 GB | 较慢 |
| **RealESRGAN** | 4 倍超分辨率 | ~200 MB | 中等 |

## 模式说明

| 模式 | 引擎 | 说明 |
|------|------|------|
| `auto` | 自动选择 | 小面积用 LaMa，大面积用 SDXL |
| `remove` | LaMa | 纯去除，只填充背景，不生成新内容 |
| `quick` | SDXL 20步 | 快速生成式填充 |
| `hq` | SDXL 30步 | 高质量生成式填充 |
| `cpu` | LaMa on CPU | 无 GPU 时使用 |

## 超分选项

| 选项 | 说明 |
|------|------|
| `none` | 不超分（默认） |
| `output` | 修复结果超分 4x |
| `input` | 先超分输入再修复（适合小图） |
| `only` | 仅超分不修复 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r lama-cleaner-plusplus/requirements.txt
pip install modelscope
```

### 2. 下载模型

```bash
python download_models.py
```

交互式菜单，按需选择：

```
Which models to download?
  1. LaMa (required, ~391 MB)
  2. SDXL (optional, ~5 GB fp16)
  3. RealESRGAN (optional, ~64 MB)
  4. All models
  0. Exit
```

也可以命令行直接指定：

```bash
python download_models.py lama        # 只下载 LaMa
python download_models.py realesrgan  # 只下载 RealESRGAN
python download_models.py sdxl        # 只下载 SDXL
python download_models.py all         # 下载全部
python download_models.py --check     # 查看模型状态
```

> 最小安装只需 LaMa（~391 MB），即可使用去除功能。

### 3. 启动

双击 `启动.bat`，或命令行：

```bash
python lama-cleaner-plusplus/app.py
```

启动后自动打开浏览器访问 `http://localhost:7860`

### 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | 端口号 | 7860 |
| `--share` | 生成公网分享链接 | False |
| `--log-level` | 日志级别 | INFO |

环境变量：

| 变量 | 说明 |
|------|------|
| `SDXL_STEPS_QUICK` | quick 模式步数 |
| `SDXL_STEPS_HQ` | hq 模式步数 |
| `SDXL_SEED` | 随机种子 |

## 使用方法

1. **上传图片** — 在左侧上传或粘贴图片
2. **涂抹区域** — 用画笔覆盖要去除的元素
3. **选模式** — 推荐用 `auto` 或 `remove` 去除水印/文字
4. **选超分** — 如需放大，选择超分选项
5. **点 Run** — 等待处理完成

### 高级参数

| 参数 | 说明 | 建议 |
|------|------|------|
| Mask 膨胀 (px) | 涂抹区域向外扩展像素数 | 水印边缘模糊时加大到 20-30 |
| Mask 羽化 (px) | mask 边缘柔化程度 | 让修复区域和周围更自然融合 |
| Prompt | 描述修复效果 | 一般留空即可 |
| Negative Prompt | 需要避免的效果 | 留空使用默认值 |

## 项目结构

```
去图像水印/
├── 启动.bat                        # 双击启动
├── download_models.py              # 模型下载脚本
├── remove_fp32_weights.py          # 清理 FP32 模型权重
├── models/                         # 本地模型目录
│   ├── lama/big_lama.pt            # LaMa 模型
│   ├── realesrgan/RealESRGAN_x4plus.pth  # RealESRGAN 模型
│   └── sdxl-inpainting/            # SDXL 模型 (diffusers 格式)
└── lama-cleaner-plusplus/
    ├── app.py                      # 入口
    ├── config.py                   # 配置
    ├── requirements.txt            # 依赖
    ├── core/
    │   ├── engine_manager.py       # 引擎管理（懒加载、互斥）
    │   ├── pipeline.py             # 修复流水线
    │   ├── service.py              # 业务逻辑（含超分编排）
    │   ├── strategy.py             # 引擎选择策略
    │   ├── mask_processor.py       # Mask 处理
    │   ├── roi.py                  # ROI 裁剪
    │   └── engines/
    │       ├── base.py             # 引擎基类
    │       ├── lama.py             # LaMa 引擎
    │       ├── sdxl.py             # SDXL 引擎
    │       └── realesrgan.py       # RealESRGAN 引擎
    ├── ui/
    │   └── gradio_app.py           # Gradio 界面
    └── utils/
        ├── gpu.py                  # GPU 显存检测
        ├── image.py                # 图像工具
        └── logger.py               # 日志
```

## 模型来源

| 模型 | 来源 | 许可证 |
|------|------|--------|
| LaMa | [ModelScope iic/cv_fft_inpainting_lama](https://www.modelscope.cn/models/iic/cv_fft_inpainting_lama) | Apache 2.0 |
| SDXL | [ModelScope AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1) | OpenRAIL-M |
| RealESRGAN | [ModelScope muse/RealESRGAN_x4plus](https://www.modelscope.cn/models/muse/RealESRGAN_x4plus) | BSD-3 |

## 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | 无（可用 CPU 模式） | RTX 3060 12GB |
| GPU 显存 | 无（纯 CPU） | 2 GB（LaMa + RealESRGAN） |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 1 GB（仅 LaMa） | 6 GB（全部模型） |

### 各功能最低显存需求

| 功能 | 所需显存 | 说明 |
|------|---------|------|
| 纯去除（LaMa） | 无（CPU） | 可选 GPU 加速，仅需 ~30MB |
| 超分辨率（RealESRGAN） | ~200 MB | 大图自动分块处理 |
| 生成式填充（SDXL quick） | ~5 GB | fp16 版本 |
| 生成式填充（SDXL hq） | ~5 GB | 步数更多，时间更长 |
| 去除 + 超分 | ~200 MB | 引擎互斥，不会同时占用 |

> 显存不足时，LaMa 和 RealESRGAN 会自动回退到 CPU 运行，SDXL 需要至少 5GB 显存。

## 常见问题

**Q: GPU 显存不足 (CUDA out of memory)**

超分和修复引擎互斥加载，不会同时占用显存。大图超分会自动分块处理。如果仍然 OOM，尝试：
- 使用 `remove` 模式代替 `quick`/`hq`（LaMa 只需 ~30MB 显存）
- 缩小输入图片尺寸

**Q: 只想去除水印，结果变成了别的东西**

选择 `remove` 模式而非 `quick`/`hq`。`remove` 使用 LaMa 引擎，只做背景填充，不会生成新内容。

**Q: 不需要 SDXL 可以只用 LaMa 吗？**

可以。SDXL 引擎是懒加载的，不安装 `diffusers` 包也不会影响 LaMa 和 RealESRGAN 的使用。
