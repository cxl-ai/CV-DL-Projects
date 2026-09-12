"""
扩展选做题: ORB vs SIFT 对比 + 参数调优 + 视频保存 + 匹配图保存 + 性能分析
"""
import sys
import os
import time
import numpy as np
import cv2
from collections import deque

# ============================================================
# 可调参数（可通过命令行或修改此处调整）
# ============================================================
PROC_SCALE   = 0.5    # 检测缩放比
DETECT_EVERY = 2      # 跳帧检测间隔
# ============================================================


def create_detector(detector_type, nfeatures):
    """根据类型创建特征检测器和匹配器"""
    if detector_type.upper() == 'ORB':
        detector = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=15,
            patchSize=31
        )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif detector_type.upper() == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")
    return detector, matcher


def detect_logo(frame, logo_small, keypoint_logo, desc_logo,
                detector, matcher, min_match, lowe_ratio, proc_scale):
    """
    在单帧中检测Logo，返回 (detected, quad_orig, inlier_matches, kp_frame, timing_ms)
    """
    t0 = time.time()
    h, w = frame.shape[:2]

    frame_small = cv2.resize(frame, None, fx=proc_scale, fy=proc_scale)
    keypoint_frame, desc_frame = detector.detectAndCompute(frame_small, None)
    kp_frame = keypoint_frame if keypoint_frame is not None else ()

    detected = False
    quad_orig = None
    inlier_matches = []

    if keypoint_frame is not None and len(keypoint_frame) > min_match:
        matches = matcher.knnMatch(desc_logo, desc_frame, k=2)

        good_matches = [m for m in matches
                        if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio]
        good_matches_first = [m[0] for m in good_matches]

        if len(good_matches_first) >= min_match:
            p0 = np.float32([keypoint_logo[m.queryIdx].pt for m in good_matches_first])
            p1 = np.float32([keypoint_frame[m.trainIdx].pt for m in good_matches_first])
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)

            if H is not None and status is not None:
                good_points = status.ravel() != 0
                inlier_count = good_points.sum()
                inlier_ratio = inlier_count / max(len(good_matches_first), 1)

                if inlier_count >= min_match and inlier_ratio >= 0.3:
                    pts = np.float32([kp.pt for kp in keypoint_logo])
                    x0s, y0s = pts.min(axis=0)
                    x1s, y1s = pts.max(axis=0)
                    quad_s = np.float32([[x0s, y0s], [x1s, y0s], [x1s, y1s], [x0s, y1s]])
                    quad_s = cv2.perspectiveTransform(
                        quad_s.reshape(1, -1, 2), H).reshape(-1, 2)
                    quad_orig = quad_s / proc_scale

                    margin = 100
                    in_bounds = np.all(
                        (quad_orig >= -margin) &
                        (quad_orig <= [w + margin, h + margin])
                    )
                    area = cv2.contourArea(quad_orig)
                    side_lens = [np.linalg.norm(quad_orig[i] - quad_orig[(i + 1) % 4])
                                 for i in range(4)]
                    shape_ok = max(side_lens) / (min(side_lens) + 1e-6) < 10

                    if in_bounds and area > 100 and shape_ok:
                        detected = True
                        inlier_matches = [good_matches_first[i]
                                          for i in range(len(good_matches_first))
                                          if good_points[i]]

    elapsed = (time.time() - t0) * 1000  # ms
    return detected, quad_orig, inlier_matches, kp_frame, elapsed


def run_detection(video_src, logo_src, detector_type, nfeatures,
                  min_match, lowe_ratio, proc_scale, detect_every,
                  save_video=False, save_matches=False, output_dir='output'):
    """
    运行完整的Logo检测流程，返回统计数据。
    """
    os.makedirs(output_dir, exist_ok=True)

    logo_orig = cv2.imread(logo_src)
    logo_small = cv2.resize(logo_orig, None, fx=proc_scale, fy=proc_scale)
    detector, matcher = create_detector(detector_type, nfeatures)
    keypoint_logo, desc_logo = detector.detectAndCompute(logo_small, None)

    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频保存
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(output_dir, f'output_{detector_type}_nf{nfeatures}_mm{min_match}_lr{int(lowe_ratio*100)}.mp4')
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w_frame, h_frame))

    frame_count = 0
    success_count = 0
    detect_times = []
    detect_buffer = deque(maxlen=5)
    last_quad = None
    last_inlier_matches = []
    last_kp_frame = ()

    prefix = f'{detector_type}_nf{nfeatures}_mm{min_match}_lr{int(lowe_ratio*100)}'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis = frame.copy()
        do_detect = (frame_count % detect_every == 0)
        raw_detected = False
        inlier_matches = []
        kp_frame_draw = last_kp_frame

        if do_detect:
            detected, quad, inlier_matches, kp_frame, elapsed = detect_logo(
                frame, logo_small, keypoint_logo, desc_logo,
                detector, matcher, min_match, lowe_ratio, proc_scale
            )
            detect_times.append(elapsed)
            kp_frame_draw = kp_frame

            if detected:
                raw_detected = True
                last_quad = quad
                last_inlier_matches = inlier_matches
                last_kp_frame = kp_frame
            else:
                last_quad = None
                last_inlier_matches = []
        else:
            if last_quad is not None:
                raw_detected = True
                inlier_matches = last_inlier_matches
                kp_frame_draw = last_kp_frame

        detect_buffer.append(raw_detected)
        stable_detected = sum(detect_buffer) >= 3

        if stable_detected and last_quad is not None:
            success_count += 1
            cv2.polylines(vis, [np.int32(last_quad)], True, (0, 255, 0), 3)

            # 保存匹配关系图
            if save_matches and do_detect:
                match_vis = cv2.drawMatches(
                    logo_small, keypoint_logo,
                    cv2.resize(vis, None, fx=proc_scale, fy=proc_scale),
                    kp_frame_draw,
                    inlier_matches, None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                img_path = os.path.join(output_dir, f'{prefix}_match_frame{frame_count:04d}.png')
                cv2.imwrite(img_path, match_vis)

        if video_writer is not None:
            video_writer.write(vis)

    cap.release()
    if video_writer is not None:
        video_writer.release()

    avg_time = np.mean(detect_times) if detect_times else 0
    detect_fps = 1000 / avg_time if avg_time > 0 else 0

    return {
        'detector': detector_type,
        'nfeatures': nfeatures,
        'min_match': min_match,
        'lowe_ratio': lowe_ratio,
        'total_frames': frame_count,
        'success_frames': success_count,
        'detection_rate': success_count / max(frame_count, 1) * 100,
        'avg_detect_time_ms': avg_time,
        'detect_fps': detect_fps,
        'total_detect_calls': len(detect_times),
    }


def print_results_table(results_list):
    """打印结果对比表格"""
    header = f"{'Detector':>8} | {'nFeat':>6} | {'MinMatch':>8} | {'Ratio':>5} | {'Total':>5} | {'Success':>7} | {'Rate':>7} | {'AvgTime':>8} | {'DetFPS':>7}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("Parameter Comparison Results")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results_list:
        print(f"{r['detector']:>8} | {r['nfeatures']:>6} | {r['min_match']:>8} | {r['lowe_ratio']:.2f} | {r['total_frames']:>5} | {r['success_frames']:>7} | {r['detection_rate']:>6.1f}% | {r['avg_detect_time_ms']:>7.1f}ms | {r['detect_fps']:>6.1f}")
    print("=" * len(header))


def print_analysis(results_list):
    """打印分析总结"""
    print("\n" + "=" * 60)
    print("Analysis: Parameter Impact on Speed & Accuracy")
    print("=" * 60)

    # 1. ORB vs SIFT 对比
    orb_results = [r for r in results_list if r['detector'] == 'ORB']
    sift_results = [r for r in results_list if r['detector'] == 'SIFT']
    if orb_results and sift_results:
        orb_best = max(orb_results, key=lambda r: r['detection_rate'])
        sift_best = max(sift_results, key=lambda r: r['detection_rate'])
        print(f"\n[ORB vs SIFT]")
        print(f"  ORB  best detection rate: {orb_best['detection_rate']:.1f}% (nfeatures={orb_best['nfeatures']}), "
              f"speed: {orb_best['detect_fps']:.1f} FPS")
        print(f"  SIFT best detection rate: {sift_best['detection_rate']:.1f}% (nfeatures={sift_best['nfeatures']}), "
              f"speed: {sift_best['detect_fps']:.1f} FPS")
        speedup = orb_best['detect_fps'] / max(sift_best['detect_fps'], 0.01)
        print(f"  ORB is ~{speedup:.1f}x faster than SIFT, "
              f"detection rate diff: {orb_best['detection_rate'] - sift_best['detection_rate']:+.1f}%")

    # 2. Lowe ratio 影响
    print(f"\n[Lowe Ratio Impact]")
    ratios = sorted(set(r['lowe_ratio'] for r in results_list))
    for ratio in ratios:
        subset = [r for r in results_list if abs(r['lowe_ratio'] - ratio) < 0.01]
        if subset:
            avg_rate = np.mean([r['detection_rate'] for r in subset])
            avg_fps = np.mean([r['detect_fps'] for r in subset])
            print(f"  ratio={ratio:.2f} -> avg detection rate: {avg_rate:.1f}%, avg detect FPS: {avg_fps:.1f}")

    # 3. MIN_MATCH_COUNT 影响
    print(f"\n[MIN_MATCH_COUNT Impact]")
    min_matches = sorted(set(r['min_match'] for r in results_list))
    for mm in min_matches:
        subset = [r for r in results_list if r['min_match'] == mm]
        if subset:
            avg_rate = np.mean([r['detection_rate'] for r in subset])
            print(f"  MIN_MATCH_COUNT={mm} -> avg detection rate: {avg_rate:.1f}%")

    # 4. nfeatures 影响
    print(f"\n[nfeatures Impact]")
    nfs = sorted(set(r['nfeatures'] for r in results_list))
    for nf in nfs:
        subset = [r for r in results_list if r['nfeatures'] == nf]
        if subset:
            avg_rate = np.mean([r['detection_rate'] for r in subset])
            avg_time = np.mean([r['avg_detect_time_ms'] for r in subset])
            print(f"  nfeatures={nf} -> avg detection rate: {avg_rate:.1f}%, avg time: {avg_time:.1f}ms")

    # 5. 综合建议
    print(f"\n[Recommendation]")
    best = max(results_list, key=lambda r: r['detection_rate'])
    print(f"  Best detection rate: {best['detector']}, nfeatures={best['nfeatures']}, "
          f"MIN_MATCH_COUNT={best['min_match']}, Lowe ratio={best['lowe_ratio']:.2f}")
    print(f"  -> Detection rate: {best['detection_rate']:.1f}%, Detect FPS: {best['detect_fps']:.1f}")

    fastest = max(results_list, key=lambda r: r['detect_fps'])
    print(f"  Fastest: {fastest['detector']}, nfeatures={fastest['nfeatures']}, "
          f"Detect FPS: {fastest['detect_fps']:.1f}")
    print("=" * 60)


if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
        logo_src = sys.argv[2]
    except Exception:
        print("Usage: python baseline_extension.py <video> <logo>")
        print("Example: python baseline_extension.py video.mp4 logo.png")
        exit(1)

    # ============================================================
    # 参数扫描配置: 多组参数对比
    # ============================================================
    configs = [
        # ── ORB vs SIFT 基础对比 ──
        {'detector': 'ORB',  'nfeatures': 5000, 'min_match': 5,  'lowe_ratio': 0.85},
        {'detector': 'SIFT', 'nfeatures': 1500, 'min_match': 5,  'lowe_ratio': 0.85},

        # ── Lowe Ratio 扫描 (固定ORB+5000+MIN=5) ──
        {'detector': 'ORB',  'nfeatures': 5000, 'min_match': 5,  'lowe_ratio': 0.70},
        {'detector': 'ORB',  'nfeatures': 5000, 'min_match': 5,  'lowe_ratio': 0.75},
        {'detector': 'ORB',  'nfeatures': 5000, 'min_match': 5,  'lowe_ratio': 0.80},

        # ── MIN_MATCH_COUNT 扫描 ──
        {'detector': 'ORB',  'nfeatures': 5000, 'min_match': 10, 'lowe_ratio': 0.85},
        {'detector': 'ORB',  'nfeatures': 5000, 'min_match': 15, 'lowe_ratio': 0.85},

        # ── nfeatures 扫描 ──
        {'detector': 'ORB',  'nfeatures': 1000, 'min_match': 5,  'lowe_ratio': 0.85},
        {'detector': 'ORB',  'nfeatures': 3000, 'min_match': 5,  'lowe_ratio': 0.85},
    ]

    all_results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {cfg['detector']}, "
              f"nfeatures={cfg['nfeatures']}, MIN_MATCH={cfg['min_match']}, "
              f"Lowe ratio={cfg['lowe_ratio']:.2f} ...")

        result = run_detection(
            video_src, logo_src,
            detector_type=cfg['detector'],
            nfeatures=cfg['nfeatures'],
            min_match=cfg['min_match'],
            lowe_ratio=cfg['lowe_ratio'],
            proc_scale=PROC_SCALE,
            detect_every=DETECT_EVERY,
            save_video=False,   # 参数扫描不保存视频（节省时间和空间）
            save_matches=False,
            output_dir='output'
        )
        all_results.append(result)
        print(f"  -> Detection rate: {result['detection_rate']:.1f}% "
              f"({result['success_frames']}/{result['total_frames']}), "
              f"avg time: {result['avg_detect_time_ms']:.1f}ms, "
              f"detect FPS: {result['detect_fps']:.1f}")

    # ── 打印汇总表格与分析 ──
    print_results_table(all_results)
    print_analysis(all_results)

    # ── 用最佳配置重新运行，保存视频和匹配关系图 ──
    best = max(all_results, key=lambda r: r['detection_rate'])
    print(f"\n>>> Re-running with best config, saving video and match images ...")
    print(f"    Config: {best['detector']}, nfeatures={best['nfeatures']}, "
          f"MIN_MATCH={best['min_match']}, Lowe ratio={best['lowe_ratio']:.2f}")

    best_result = run_detection(
        video_src, logo_src,
        detector_type=best['detector'],
        nfeatures=best['nfeatures'],
        min_match=best['min_match'],
        lowe_ratio=best['lowe_ratio'],
        proc_scale=PROC_SCALE,
        detect_every=DETECT_EVERY,
        save_video=True,
        save_matches=True,
        output_dir='output'
    )
    print(f"    Output video and match images saved to output/")
    print(f"    Detection rate: {best_result['detection_rate']:.1f}%, "
          f"Detect FPS: {best_result['detect_fps']:.1f}")
