import cv2
import operator
import numpy as np


def show_image(img, win='image'):
    """显示图片，直到按下任意键继续"""
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_digits(digits, color=255, withBorder=True):
    """将提取并处理过的81个单元格图片构成的列表显示为二维9*9大图"""
    rows = []
    if withBorder:
        # 给复制图片（每个数字图片）四周向外加边框，此处上下左右的边框宽度均为1像素，且默认加白色边框
        with_border = [cv2.copyMakeBorder(digit, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, color) for digit in digits]
    for i in range(9):  # 一行行地处理
        if withBorder:  # axis=0 表示垂直方向拼接，axis=1 表示水平方向拼接
            row = np.concatenate(with_border[i * 9: (i + 1) * 9], axis=1)  # 水平方向拼接9张图片，得到一行图片
        else:
            row = np.concatenate(digits[i * 9: (i + 1) * 9], axis=1)
        rows.append(row)
    bigImage = np.concatenate(rows, axis=0)  # 若每个数字图片尺寸为58*58，则加边框后单张数字图片大小为60*60
    show_image(bigImage, 'bigImage')  # 其中60=58+2，此时bigImage尺寸为 540*540
    cv2.imwrite('segmentedBigImg.jpg', bigImage)


def convert_with_color(color, img):
    """如果color是元组且img是灰度图，则动态地转换img为彩图"""
    # if len(color) == 3:
    #     if len(img.shape) == 2:  # (h,w)
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     elif len(img.shape) == 3 and img.shape[2] == 1:  # 单通道 (h, w, 1)
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 处理单通道或二维图像
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img  # 三通道彩图


def pre_process_gray(gray, skip_dilate=False):  # gray:灰度图
    """使用高斯模糊、自适应阈值分割和/或膨胀来暴露图像的主特征"""
    proc = cv2.GaussianBlur(gray.copy(), (9, 9), 0)  # proc： 降噪后的灰度图
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # 处理完毕的proc：二值图，前景数字是白色

    if not skip_dilate:
        # 膨胀的目的有二：消除前景数字内部的小孔洞；增大格子边线的尺寸，使之更明显。
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)  # 结构元
        proc = cv2.dilate(proc, kernel)
    return proc  # proc：二值图，前景数字是白色，背景是黑色


def display_points(in_img, points, radius=5, color=(0, 0, 255)):
    """在图像上绘制彩色圆点，原图像可能是灰度图"""
    img = in_img.copy()  # 在二值化单通道图片上画不了三通道红色， 所以将要转换为BGR三通道
    img = convert_with_color(color, img)  # 如有必要，动态转换为彩图
    for point in points:
        cv2.circle(img, tuple(int(x) for x in point), radius, color, -1)
    return img


def find_corners_of_largest_polygon(bin_img):  # bin_img：二值图，前景数字是白色
    """找出图像中面积最大轮廓的4个角点。"""
    contours, h = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours中每个点坐标都是二维数组
    # print(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积降序排序
    polygon = contours[0]  # Largest polygon
    print(polygon.shape)  # 输出：(193, 1, 2)

    # 右下角点具有最大的（x+y）值,左上角点具有最小的（x+y）值,左下角点具有最小的（x-y）值,右上角点具有最大的（x-y）值
    # bottom_right_idx: 右下角点的编号，_: （x+y）的最大值，但我们不关心
    bottom_right_idx, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))  # enumerate([pt[0][0] + pt[0][1] for pt in polygon]) 生成带有索引和x+y值的序列
    top_left_idx, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))  # key=operator.itemgetter(1)表示使用元组中的第二个元素进行比较
    bottom_left_idx, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right_idx, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # 返回四边形的四个角点坐标数据
    # 每个点的坐标示例如：[[481 68]], 可参考第四章第三次课上代码
    points = [polygon[top_left_idx][0], polygon[top_right_idx][0],
              polygon[bottom_right_idx][0], polygon[bottom_left_idx][0]]
    show_image(display_points(bin_img, points), '4-points')
    return points  # 顺时针方式安排


def distance_between(p1, p2):
    """返回两点之间的标量距离"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(gray, crop_rect):
    """将灰度图像中由4角点围成的四边形区域裁剪出来，并将其扭曲为类似大小的正方形"""
    # cv2.warpAffine()不能处理视场和图像不平行的问题！ 仿射变换getAffineTransform只需要3个点构建，透视变换getPerspectiveTransform则需要找到4个点构建kernel
    # 四边形的四个顶点坐标
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    # print(top_left.dtype)  # [54 63], int32

    # 封装为列表，并把数据类型显式转换为float32，否则后面的 `getPerspectiveTransform` 将抛出错误
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # 获取四边形的最大边长
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # 描述一个边长为计算长度side的正方形，这就是我们的变换目标
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], dtype='float32')  # 目标方形（顺时针方式组织）

    # 通过比较前后4个点，获取用于扭曲图像以适合正方形的透视变换矩阵（3*3矩阵）
    m = cv2.getPerspectiveTransform(src, dst)

    # 因为src是 img的局部点集，所以下面这句只是把局部图像变为正方形，而不是整体！
    cropped = cv2.warpPerspective(gray, m, (int(side), int(side)))  # 目标正方形的宽和高
    show_image(cropped, 'cropped')
    cv2.imwrite('cropped.png', cropped)
    return cropped


def infer_grid(square_gray):
    """从正方形灰度图像推断其内部81个单元网格的位置（以等分方式）。"""
    squares = []
    side = square_gray.shape[:1]  # 大正方形的高度即边长 (366,)
    side = side[0] / 9  # 小正方形的高度 40.666666666666664

    # 从左到右一行行读
    for j in range(9):  # j: 行号（y坐标）
        for i in range(9):  # i：列号（x坐标） 每次获取一个单元格的左上角点坐标p1和右下角点坐标p2
            p1 = (i * side, j * side)  # 包围盒的左上角点
            p2 = ((i + 1) * side, (j + 1) * side)  # 包围盒的右下角点
            squares.append((p1, p2))  # (p1, p2) 代表一个单元格的位置和大小
    return squares  # len(squares) = 81


def cut_from_rect(img, rect):
    """从图像中切出一个矩形ROI区域。注意先y后x，即先h后w"""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]  # get part square roi of img


def scale_and_centre(img, size, margin=0, background=0):  # img：切出的主特征图片， size边长 margin两侧边距和(偶数) background边框像素值
    """把单元格图片img经缩放且加边距，置于边长为size的新背景正方形图像中"""
    h, w = img.shape[:2]

    def centre_pad(length):  # length是切出图的宽度或者高度
        padAll = size - length   # padAll是双向边距和
        if padAll % 2 == 0:
            pad1 = int(padAll / 2)  # 整除
            pad2 = pad1
        else:
            pad1 = int(padAll / 2)  # 不能整除
            pad2 = pad1 + 1
        return pad1, pad2

    def scale(r, x):
        return int(r * x)

    if h > w:  # 以高度为主做居中操作
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h  # (size - margin)：经缩放之后的切出图的目标高度
        w, h = scale(ratio, w), scale(ratio, h)  # 计算切出图缩放之后的宽和高（其实高度h在上一行代码已确定）
        l_pad, r_pad = centre_pad(w)
    else:  # 以宽度为主做居中操作
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w  # (size - margin)：经缩放之后的切出图的目标宽度
        w, h = scale(ratio, w), scale(ratio, h)  # 计算切出图缩放之后的宽和高（其实宽度w在上一行代码已确定）
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))  # 对切出图做缩放，使它变成w宽h高
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    if margin % 2 != 0:  # 如果双向边距margin不是偶数，则经过上述算法处理后，img的宽度高度中必有其一不是size。
        img = cv2.resize(img, (size, size))
    return img


def find_largest_feature(inp_img, scan_tl, scan_br):  # 主特征如果存在，它一定会与位于scan_tl, scan_br范围内的种子点连通
    """利用floodFill函数返回它所填充区域的边界框的事实，找到图像中的主特征，将此结构填充为白色，其余部分降为黑色。"""
    img = inp_img.copy()
    h, w = img.shape[:2]
    max_area = 0
    seed_point = (None, None)

    for x in range(scan_tl[0], scan_br[0]):  # 水平方向扫描范围
        for y in range(scan_tl[1], scan_br[1]):  # 垂直方向扫描范围
            if img.item(y, x) == 255 and x < w and y < h:  # 注意 .item()方法中参数顺序为 y, x，因为图像的索引方式是先行后列
                area = cv2.floodFill(img, None, (x, y), 64)  # 将与种子点相连接的注水区域换成特定的颜色64
                print('area', area) # area第一个参数表示像素值，第二个参数是填充后的图像，第三个参数是mask，第四个参数是填充区域矩形框左上角坐标，以及w,h
                if area[0] > max_area:  # 更新max_area像素值
                    max_area = area[0]
                    seed_point = (x, y)
    for x in range(w):  # 前面的代码仅在中间的一个小区域内扫描，对于前景像素的处理可能有遗漏
        for y in range(h):
            if img.item(y, x) == 255 and x < w and y < h:
                cv2.floodFill(img, None, (x, y), 64)  # 将剩余的前景像素替换为灰色64

    if all([p is not None for p in seed_point]): # 判断种子点x,y都存在，即主特征存在
        cv2.floodFill(img, None, seed_point, 255)  # 主特征恢复为前景颜色255， 此时非主特征像素还是保持64
    top, bottom, left, right = h, 0, w, 0
    for x in range(w):
        for y in range(h):
            if img.item(y, x) == 64:
                cv2.floodFill(img, None, (x, y), 0)  # 遍历整幅图像，将非主特征归为背景0
            # 不断更新主特征的包围矩形
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(bin_img, rect, size):  # bin_img：二值图, rect:某格子的位置
    """从预处理后的二值方形大格子图中提取由rect指定的小单元格数字图"""
    digit = cut_from_rect(bin_img, rect)
    # show_image(digit, 'digit')
    # 使用漫水填充法来获得经过盒子中间的最大特征
    # margin（边距） 用于定义中间的一个区域，我们期待该区域的某个像素属于最大特征。
    h, w = digit.shape[:2]  # 41 40 or 41 41
    # 边长: np.mean([h, w])  # 40.5 or 41, 边长一半: int(np.mean([h, w]) / 2)
    margin = int(np.mean([h, w]) / 2.5)  # margin：边距，比边长一半要小一些

    # digit_roi = digit[margin: h-margin, margin: w-margin]
    # show_image(digit_roi, 'digit_roi')
    # 从中间区域开始寻找主特征, 返回值bbox是围住最大特征的矩形（（left, top）,(right, bottom))
    flooded, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])  # 扫描起始点左上角：scan_tl（top-left） 扫描最终点右下角：scan_br(below-right)

    # 计算紧凑数字图的宽和高
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    if w > 0 and h > 0 and (w * h) > 200:  # 若是主特征，则h*w肯定大于200
        digit = cut_from_rect(flooded, bbox)  # 注意：flooded已去掉了非主特征
        return scale_and_centre(digit, size, 4)  # margin最好写偶数
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(square_gray, squares, size):  # square_gray：灰度方形大格子图，squares：81个小单元格的位置
    """提取小单元格数字，组织成数组形式"""
    digits = []
    square_bin = pre_process_gray(square_gray.copy(), skip_dilate=True)  # 灰度方形大格子图一样需要预处理

    # 用红色画出水平垂直分割线，用于目测各单元格的主特征是否保持相对完好
    color = convert_with_color((0, 0, 255), square_bin)
    h, w = color.shape[:2]  # (366, 366)
    for i in range(10):
        cv2.line(color, (0, int(i * h / 9)), (w - 1, int(i * h / 9)), (0, 0, 255))  # 10条水平分割线
        cv2.line(color, (int(i * w / 9), 0), (int(i * w / 9), h - 1), (0, 0, 255))  # 10条垂直分割线
    # cv2.imshow('drawRedLine', color)
    # cv2.waitKey(1)
    show_image(color, 'drawRedLine')

    for square in squares:
        digits.append(extract_digit(square_bin, square, size))  # square_bin：二值图， square是小格子左上角和右下角坐标，size = 58
    return digits


def parse_grid(path):
    original = cv2.imread(path, 0)  # original：灰度图
    processed = pre_process_gray(original)  # processed：二值图，前景数字是白色
    corners = find_corners_of_largest_polygon(processed)  # corners: 顺时针组织的四个大格子角点
    cropped = crop_and_warp(original, corners)  # 从灰度图切出大格子图，然后矫正
    squares = infer_grid(cropped)  # 获取81个小正方形
    digits = get_digits(cropped, squares, 58)  # 灰度矫正图
    show_digits(digits, withBorder=True)  # 若withBorder= False，则单元格无白色边框


if __name__ == '__main__':
    parse_grid('../cvimages/ch06/sudoku.png')
