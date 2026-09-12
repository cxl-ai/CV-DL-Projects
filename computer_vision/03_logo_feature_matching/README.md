# 03 - 图像特征点检测与品牌 Logo 空间定位

## 项目简介
本项目针对真实视频场景中光照剧烈变化、尺度缩放、视角倾斜及部分遮挡等复杂工况，实现了对特定品牌 Logo 的高精度检测、匹配与空间轮廓定位。对比了 SIFT、ORB 等经典特征提取算子，并利用 RANSAC 单应性矩阵映射（Homography）实现了动态视频帧中的精确目标框定。

---

## 核心技术点
1. **特征提取算法对比**：
   - **SIFT（尺度不变特征变换）**：对尺度缩放、图像旋转、仿射变换和光照变化具有极强鲁棒性，特征向量为 128 维。
   - **ORB（Oriented FAST and Rotated BRIEF）**：计算效率高，适合高帧率实时场景。
2. **特征匹配与误匹配剔除**：
   - 使用 **FLANN（快速近似最近邻）** 与 **BFMatcher（暴力匹配）**；
   - 引入 **Lowe 提出的比值测试（Lowe's Ratio Test）**，有效过滤 80% 以上模糊伪匹配。
3. **单应性矩阵与空间几何定位**：
   - 利用 `cv2.findHomography` 结合 **RANSAC 算法** 剔除外点，解算模板图到当前帧的单应性投影矩阵 $H$；
   - 利用 `cv2.perspectiveTransform` 计算模板四个角点在当前视频帧中的透视位置，绘制四边形边界框。
4. **扩展优化（Extension）**：
   - 多尺度模板金字塔搜索；
   - 帧间平滑与时序追踪，消除视频单帧抖动。

---

## 文件结构
```text
03_logo_feature_matching/
├── README.md                      # 本说明文档
├── baseline_improved.py           # 改进版 SIFT + FLANN + RANSAC 核心定位代码
├── baseline_extension.py          # 扩展性能调优与多模板对比测试代码
├── 第5章 综合项目课上代码 baseline.py# 初始 Baseline 实现
├── 2b327b4fdaf69662ea70198ad29b2c1b.png # 品牌 Logo 模板图像
├── video.mp4                      # 测试输入视频
└── output/                        # 视频推理输出与特征匹配过程帧
```

---

## 快速运行
```bash
# 运行优化版算法对视频进行检测定位
python baseline_improved.py

# 运行性能扩展测试与算法指标分析
python baseline_extension.py
```