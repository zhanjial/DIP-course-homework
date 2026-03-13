import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image_orig = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image_orig, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image_orig = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image_orig
    if image_orig is None:
        return None

    x, y = evt.index[0], evt.index[1]
    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image_orig.copy()
    for i in range(len(points_src)):
        p_s = tuple(map(int, points_src[i]))
        cv2.circle(marked_image, p_s, 5, (255, 0, 0), -1)  # 蓝色
        if i < len(points_dst):
            p_d = tuple(map(int, points_dst[i]))
            cv2.circle(marked_image, p_d, 5, (0, 0, 255), -1)  # 红色
            cv2.arrowedLine(marked_image, p_s, p_d, (0, 255, 0), 2, tipLength=0.3)
    return marked_image

#使用的是论文中移动最小二乘的图像刚性变形算法
def point_guided_deformation(image, p, q, alpha=2.0):
    """
    p: 目标位置 (Target points)
    q: 原始位置 (Source points)
    """
    h, w, _ = image.shape
    p = np.array(p, dtype=np.float32)
    q = np.array(q, dtype=np.float32)

    # 生成网格坐标
    vx, vy = np.meshgrid(np.arange(w), np.arange(h))
    v = np.stack([vx, vy], axis=-1).reshape(-1, 2).astype(np.float32)

    # 1. 计算权重 (基于目标点 p)
    # dists shape: (N_pixels, N_control_pts)
    diff = v[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2) #计算欧氏距离平方
    weights = 1.0 / (dists + 1e-6) ** alpha #计算权重，距离越近，权重越大
    weights /= np.sum(weights, axis=1, keepdims=True) #归一化，使每个像素的权重和为1

    # 2. 计算加权中心
    p_star = np.sum(weights[:, :, np.newaxis] * p[np.newaxis, :, :], axis=1)#np包广播数组的功能
    q_star = np.sum(weights[:, :, np.newaxis] * q[np.newaxis, :, :], axis=1)

    p_hat = p[np.newaxis, :, :] - p_star[:, np.newaxis, :]
    q_hat = q[np.newaxis, :, :] - q_star[:, np.newaxis, :]

    # 3. 计算刚性变换 (Rigid Transformation)
    # 构造垂直向量
    p_hat_perp = np.stack([-p_hat[:, :, 1], p_hat[:, :, 0]], axis=-1)
    v_minus_pstar = v - p_star
    v_minus_pstar_perp = np.stack([-v_minus_pstar[:, 1], v_minus_pstar[:, 0]], axis=-1)

    # 计算 mu (模的平方和)
    mu = np.sum(weights * np.sum(p_hat ** 2, axis=2), axis=1)
    mu = np.maximum(mu, 1e-6)

    # 核心映射函数 f(v)
    f_v = np.zeros_like(v)
    for i in range(len(p)):
        s1 = np.sum(p_hat[:, i, :] * v_minus_pstar, axis=1)
        s2 = np.sum(p_hat_perp[:, i, :] * v_minus_pstar, axis=1)

        weight_factor = (weights[:, i] / mu)[:, np.newaxis]

        # 映射到原图对应的坐标
        f_v += weight_factor * (q_hat[:, i, :] * s1[:, np.newaxis] +
                                np.stack([-q_hat[:, i, 1], q_hat[:, i, 0]], axis=-1) * s2[:, np.newaxis])

    f_v += q_star

    # 4. 映射与插值
    map_x = np.clip(f_v[:, 0].reshape(h, w), 0, w - 1).astype(np.float32)
    map_y = np.clip(f_v[:, 1].reshape(h, w), 0, h - 1).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def run_warping(alpha_val):
    global points_src, points_dst, image_orig
    # 只有成对的点才能计算
    num_pairs = min(len(points_src), len(points_dst))
    if image_orig is None or num_pairs < 1:
        return image_orig

    # 关键逻辑：计算 [目标点 -> 原始点] 的映射
    result = point_guided_deformation(image_orig, points_dst[:num_pairs], points_src[:num_pairs], alpha=alpha_val)
    return result


def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image_orig


# --- Gradio 界面 ---
with gr.Blocks(title="Point Guided Deformation") as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="1. 上传图片", interactive=True)
            point_canvas = gr.Image(label="2. 标注控制点 (原地双击固定)")
            alpha_slider = gr.Slider(minimum=1.0, maximum=4.0, value=2.0, step=0.1, label="变形平滑度 (Alpha)")
        with gr.Column():
            result_img = gr.Image(label="3. 变形结果")

    with gr.Row():
        run_btn = gr.Button("Run Warping", variant="primary")
        clear_btn = gr.Button("Clear Points")

    input_img.upload(upload_image, input_img, point_canvas)
    point_canvas.select(record_points, None, point_canvas)
    run_btn.click(run_warping, inputs=[alpha_slider], outputs=result_img)
    clear_btn.click(clear_points, None, point_canvas)

demo.launch()