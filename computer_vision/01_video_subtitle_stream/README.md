# 01 - 视频流处理与动态滚动字幕弹幕系统

## 项目简介
本项目基于 OpenCV 与 PIL 实现了对视频流的实时帧级解析、中文文字渲染、由下至上的平滑字幕滚动动画以及双轨弹幕叠加合成。解决了 OpenCV 无法原生渲染高质量抗锯齿中文字体的问题，实现了影视级字幕与弹幕效果。

---

## 核心技术点
1. **视频流逐帧捕获与编解码**：使用 `cv2.VideoCapture` 与 `cv2.VideoWriter` 维护帧率（FPS）、分辨率与色彩空间。
2. **高质量中文字体渲染**：在 OpenCV 图像数组与 PIL Image 之间进行双向转换，利用 `ImageFont` 实现 TrueType 字体抗锯齿绘制。
3. **坐标位移插值算法**：计算时间帧与 y 坐标线性偏移，控制滚动字幕与横向弹幕的飞行速度与出现时序。

---

## 文件结构
```text
01_video_subtitle_stream/
├── README.md                      # 本说明文档
├── welcome.mp4                    # 输入/输出演示测试视频
└── 第二次作业空模板-制作视频.ipynb   # 交互式 Jupyter Notebook 实现与源码
```

---

## 快速运行
建议使用 Jupyter Lab 或 VSCode 打开并运行 Notebook：
```bash
jupyter notebook 第二次作业空模板-制作视频.ipynb
```