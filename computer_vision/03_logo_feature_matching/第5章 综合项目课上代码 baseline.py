import sys
import os
import numpy as np
import cv2


def show_frame(img, title = 'point match'):
    cv2.imshow(title, img)    # bug: 没有调节显示窗口的大小至合适的比例
    cv2.waitKey(1)


# 最少可信匹配特征点数量
MIN_MATCH_COUNT = 10
index = 0

# Step 1: 获取命令行输入
if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
        logo_src = sys.argv[2]
    except:
        print("请输入视频路径和logo图片路径")
        exit(1)

    os.makedirs('results', exist_ok=True)

    # Step 2: 初始化特征点检测方法ORB
    detector = cv2.ORB_create(nfeatures=1000)

    # 初始化特征匹配方法flann
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    flann_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    # search_params = dict()
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug {}: need to pass empty dict (#1329)

    # Step 3: logo图片读取和计算特征点
    logo = cv2.imread(logo_src)
    keypoint_logo, desc_logo = detector.detectAndCompute(logo, None)

    # Step 4: 视频读取和特征点匹配
    cap = cv2.VideoCapture(video_src)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_cur = frame.copy()
        vis = frame.copy()

        # 计算视频帧特征描述符
        keypoint_frame, desc_frame = detector.detectAndCompute(frame_cur, None)
        # 特征点匹配
        matches = matcher.knnMatch(desc_logo, desc_frame, k=2)

        if len(keypoint_frame) <= MIN_MATCH_COUNT:
            continue

        # Step 5: 提取可信的匹配特征对
        good_matches = [m for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.7]
        good_matches_first = [m[0] for m in good_matches]

        if len(good_matches_first) < MIN_MATCH_COUNT:
            continue

        # Step 6: 计算匹配特征对之间的透视变换
        p0 = [keypoint_logo[m.queryIdx].pt for m in good_matches_first]
        p1 = [keypoint_frame[n.trainIdx].pt for n in good_matches_first]  # bug m
        p0, p1 = np.float32((p0, p1))
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        # 计算符合透视变换的匹配点个数
        good_points = status.ravel() != 0
        if good_points.sum() < MIN_MATCH_COUNT:
            continue

        # Step 7:计算logo图片的特征点包围盒
        min_x = 0
        min_y = 0
        max_x = logo.shape[1]
        max_y = logo.shape[0]
        for i in range(len(keypoint_logo)):
            min_x = np.min((min_x, keypoint_logo[i].pt[0]))
            min_y = np.min((min_y, keypoint_logo[i].pt[1]))
            max_x = np.max((max_x, keypoint_logo[i].pt[0]))
            max_y = np.max((max_y, keypoint_logo[i].pt[1]))
        logo_rect = (min_x, min_y, max_x, max_y)

        # 根据透视变换矩阵画出变换后的包围盒
        x0, y0, x1, y1 = logo_rect
        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

        cv2.polylines(vis, [np.int32(quad)], True, (255, 255, 255), 2)

        show_frame(vis)

        # 是否将识别出的图片保存至results文件夹中？
        index += 1
        save_path = f'results/{index}.png'
        ok = cv2.imwrite(save_path, vis)
        if ok:
            print('已保存：', save_path)
        else:
            print('保存失败：', save_path)

cap.release()
cv2.destroyAllWindows()