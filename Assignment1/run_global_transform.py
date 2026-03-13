import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication ：2×3 的仿射变换矩阵 转换为 3×3 的齐次坐标矩阵
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)


    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    h,w = image.shape[:2]
    center_x, center_y = w/2, h/2 #获取填充后的图像中心
    M1 = np.array([[1, 0, -center_x], [0, 1, -center_y]], dtype=np.float32) #将图片平移到原点
    M2 = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)  # 缩放
    M3 = cv2.getRotationMatrix2D((0, 0), rotation, 1.0)  # 旋转（绕原点）
    M4 = np.array([[1, 0, center_x], [0, 1, center_y]], dtype=np.float32)  # 平移回
    M5 = np.array([[1, 0, translation_x], [0, 1, translation_y]], dtype=np.float32)  # 用户平移
    #用前面的函数将矩阵辐射成3×3的矩阵
    M1_3x3 = to_3x3(M1)
    M2_3x3 = to_3x3(M2)
    M3_3x3 = to_3x3(M3)
    M4_3x3 = to_3x3(M4)
    M5_3x3 = to_3x3(M5)
    #将所有要进行的变化相乘
    M_final = M5_3x3 @ M4_3x3 @ M3_3x3 @ M2_3x3 @ M1_3x3
    M_final = M_final[:2, :]
    #将这些变化应用到图像上
    transformed_image = cv2.warpAffine(image, M_final, (w, h))
    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
