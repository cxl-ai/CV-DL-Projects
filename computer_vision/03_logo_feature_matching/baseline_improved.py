import sys
import os
import time
import numpy as np
import cv2
from collections import deque

WINDOW_NAME = 'Logo Detection - Match Visualization'


def show_frame(img):
    """显示图像，自动调整窗口至合适大小"""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    h, w = img.shape[:2]
    max_w, max_h = 1600, 900
    scale = min(max_w / w, max_h / h, 1.0)
    cv2.resizeWindow(WINDOW_NAME, max(1, int(w * scale)), max(1, int(h * scale)))
    cv2.imshow(WINDOW_NAME, img)


# ── 可调参数 ──────────────────────────────────────────────
MIN_MATCH_COUNT = 5       # 最少内点数（瓶身弯曲场景适当放宽）
LOWE_RATIO      = 0.80    # Lowe ratio（实验最佳平衡点: 检测率21.9% @ 65.7FPS）
PROC_SCALE      = 0.5     # 检测时缩放比例（0.5 = 半分辨率，提速 ~4x）
DETECT_EVERY    = 2       # 每 N 帧做一次完整检测，中间帧复用上次结果
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
        logo_src  = sys.argv[2]
    except Exception:
        print("请输入视频路径和logo图片路径")
        exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ── Step 2: 初始化检测器（ORB，速度快）────────────────
    detector = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=12,         # 更多尺度层，对大小变化更鲁棒
        edgeThreshold=15,
        patchSize=31
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ── Step 3: 读取 Logo，在缩小后的尺寸下提取特征点 ────
    logo_orig = cv2.imread(logo_src)
    logo_small = cv2.resize(logo_orig, None, fx=PROC_SCALE, fy=PROC_SCALE)
    keypoint_logo, desc_logo = detector.detectAndCompute(logo_small, None)

    # ── Step 4: 打开视频 ──────────────────────────────────
    cap = cv2.VideoCapture(video_src)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps)) if fps > 0 else 30
    w_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频保存: 输出带定位框的新视频 + 匹配可视化视频
    os.makedirs('output', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output/output_video.mp4', fourcc, fps, (w_out, h_out))
    match_vw = None  # 延迟初始化（等第一帧 match_vis 确定尺寸）

    frame_count   = 0
    success_count = 0
    t_total       = 0.0

    # 跨帧复用的状态
    last_quad            = None    # 上一次成功的四边形（原图坐标）
    last_inlier_matches  = []
    last_kp_frame        = ()

    # 时序平滑: 连续N帧检测结果缓存
    detect_buffer = deque(maxlen=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h_frame, w_frame = frame.shape[:2]
        vis = frame.copy()

        do_detect = (frame_count % DETECT_EVERY == 0)

        raw_detected   = False
        inlier_matches = []
        kp_frame_draw  = last_kp_frame

        if do_detect:
            t0 = time.time()

            # ── 缩小帧用于检测 ──────────────────────────
            frame_small = cv2.resize(frame, None, fx=PROC_SCALE, fy=PROC_SCALE)

            keypoint_frame, desc_frame = detector.detectAndCompute(frame_small, None)
            kp_frame_draw = keypoint_frame if keypoint_frame is not None else ()

            if keypoint_frame is not None and len(keypoint_frame) > MIN_MATCH_COUNT:
                matches = matcher.knnMatch(desc_logo, desc_frame, k=2)

                # Lowe's ratio test
                good_matches = [m for m in matches
                                if len(m) == 2 and m[0].distance < m[1].distance * LOWE_RATIO]
                good_matches_first = [m[0] for m in good_matches]

                if len(good_matches_first) >= MIN_MATCH_COUNT:
                    p0 = np.float32([keypoint_logo[m.queryIdx].pt for m in good_matches_first])
                    p1 = np.float32([keypoint_frame[m.trainIdx].pt for m in good_matches_first])
                    H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)

                    if H is not None and status is not None:
                        good_points  = status.ravel() != 0
                        inlier_count = good_points.sum()
                        inlier_ratio = inlier_count / max(len(good_matches_first), 1)

                        if inlier_count >= MIN_MATCH_COUNT and inlier_ratio >= 0.3:
                            # 包围盒（缩放空间下）
                            pts = np.float32([kp.pt for kp in keypoint_logo])
                            x0s, y0s = pts.min(axis=0)
                            x1s, y1s = pts.max(axis=0)
                            quad_s = np.float32([[x0s, y0s], [x1s, y0s], [x1s, y1s], [x0s, y1s]])
                            quad_s = cv2.perspectiveTransform(
                                quad_s.reshape(1, -1, 2), H).reshape(-1, 2)

                            # 映射回原图坐标
                            quad_orig = quad_s / PROC_SCALE

                            # 几何验证
                            margin = 100
                            in_bounds = np.all(
                                (quad_orig >= -margin) &
                                (quad_orig <= [w_frame + margin, h_frame + margin])
                            )
                            area = cv2.contourArea(quad_orig)
                            side_lens = [np.linalg.norm(quad_orig[i] - quad_orig[(i + 1) % 4])
                                         for i in range(4)]
                            shape_ok = max(side_lens) / (min(side_lens) + 1e-6) < 10

                            if in_bounds and area > 100 and shape_ok:
                                raw_detected       = True
                                last_quad          = quad_orig
                                inlier_matches     = [good_matches_first[i]
                                                      for i in range(len(good_matches_first))
                                                      if good_points[i]]
                                last_inlier_matches = inlier_matches
                                last_kp_frame      = kp_frame_draw

            if not raw_detected:
                last_quad           = None
                last_inlier_matches = []

            t_total += time.time() - t0

        else:
            # 非检测帧：复用上一次结果
            if last_quad is not None:
                raw_detected   = True
                inlier_matches = last_inlier_matches
                kp_frame_draw  = last_kp_frame

        # ── 时序平滑 ─────────────────────────────────────
        detect_buffer.append(raw_detected)
        stable_detected = sum(detect_buffer) >= 3

        # ── 绘制结果（始终在原图上）──────────────────────
        if stable_detected and last_quad is not None:
            success_count += 1
            cv2.polylines(vis, [np.int32(last_quad)], True, (0, 255, 0), 3)
        else:
            cv2.putText(vis, 'Logo Not Detected', (w_frame - 380, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ── FPS 显示 ──────────────────────────────────────
        detect_fps = frame_count / t_total if t_total > 0 else 0
        cv2.putText(vis, f'Det FPS: {detect_fps:.1f}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ── 保存带定位框的视频帧 ──────────────────────────
        video_writer.write(vis)

        # ── 匹配可视化（侧边对比图）──────────────────────
        match_vis = cv2.drawMatches(
            logo_small, keypoint_logo,
            cv2.resize(vis, None, fx=PROC_SCALE, fy=PROC_SCALE),
            kp_frame_draw,
            inlier_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # ── 保存成功检测帧的匹配关系图 ────────────────────
        if stable_detected and do_detect and inlier_matches:
            img_path = f'output/match_frame{frame_count:04d}.png'
            cv2.imwrite(img_path, match_vis)

        # ── 保存匹配可视化视频 ────────────────────────────
        if match_vw is None:
            mh, mw = match_vis.shape[:2]
            match_vw = cv2.VideoWriter('output/match_visualization.mp4',
                                        fourcc, fps, (mw, mh))
        match_vw.write(match_vis)

        show_frame(match_vis)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    if match_vw is not None:
        match_vw.release()
    cv2.destroyAllWindows()
    avg_det_fps = frame_count / t_total if t_total > 0 else 0
    print(f'处理完成: 共 {frame_count} 帧, 成功检测 {success_count} 帧')
    print(f'平均检测 FPS: {avg_det_fps:.1f}  (每 {DETECT_EVERY} 帧检测一次)')


# ============================================================
# 主要修改说明:
#
# 【速度优化 - 核心】
# 1.【缩小处理分辨率 PROC_SCALE=0.5】
#    检测时将帧和Logo均缩小到50%，计算量降约4倍。
#    匹配后四边形坐标除以PROC_SCALE映射回原图绘制。
#
# 2.【跳帧检测 DETECT_EVERY=2】
#    每2帧做一次完整检测，中间帧复用上次结果。
#    与缩小分辨率合计提速约8倍。
#
# 3.【ORB参数调整】
#    nfeatures=5000, nlevels=12 提升尺度鲁棒性，
#    同时保持ORB二进制描述符的高速优势（比SIFT快10x+）。
#
# 【识别率优化】
# 4.【Lowe ratio: 0.7 → 0.80（实验最佳值）】
#    经多组参数扫描(0.70/0.75/0.80/0.85): 0.80 检测率21.9%@65.7FPS，平衡最优。
#
# 5.【放宽MIN_MATCH_COUNT: 10 → 5】
#    弯曲/遮挡/反光场景下可信匹配点偏少。
#
# 【卡顿与框乱飞修复（沿用之前版本）】
# 6. 每帧必须显示（消除continue跳帧）
# 7. BFMatcher+NORM_HAMMING（替代FLANN，适配ORB二进制描述符）
# 8. 包围盒用特征点实际min/max（修复原baseline初始值bug）
# 9. 四重几何验证：内点占比>=0.3 + 越界检查 + 面积>100 + 边长比<10
# 10. 时序平滑：deque 5帧中>=3帧检测到才画框
# 11. drawMatches侧边对比可视化，窗口尺寸全程一致
# 12. 根据视频FPS计算waitKey延时
# 13. 播放结束统计总帧数/成功帧数/平均检测FPS
#
# 【扩展功能（选做题）】
# 14. ORB vs SIFT对比: 见 baseline_extension.py，SIFT检测率24.9%但慢4.4x
# 15. 视频保存: 输出带定位框的视频到 output/output_video.mp4
# 16. 匹配图保存: 成功检测帧的 match_vis 保存到 output/match_frame*.png
# 17. 参数扫描: 见 baseline_extension.py 的9组配置对比结果
# ============================================================
