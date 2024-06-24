import gradio as gr
from gradio_image_prompter import ImagePrompter
import cv2 as cv
import numpy as np

from Grabcut import GrabCut

DRAW_BG = 0
DRAW_FG = 1


thickness = 3  # 画笔粗细

css = """
.custom-foreground-point {
    background-color: #00FF00 !important;  /* 绿色 */
}
.custom-background-point {
    background-color: #FF0000 !important;  /* 红色 */
}
"""

def process_input(data, mode,thickness, state):
    img = np.array(data['image'])
    new_points = data['points']
    if state is None:
        # 如果 state 为空，则初始化
        state = {
            "img": img.copy(),
            "mask": np.zeros(img.shape[:2], dtype=np.uint8),
            "gc": None,
            "rect": (0, 0, 1, 1),
            "points": None,
        }

    img2 = state["img"]
    mask = state["mask"]
    gc = state["gc"]
    rect = state["rect"]
    points = state["points"]
    if mode == "矩形选框":
        # 每次矩形选框时重置 GrabCut 模型和掩码
        for prompt in new_points:
            x1, y1, x2, y2 = int(prompt[0]), int(prompt[1]), int(prompt[3]), int(prompt[4])
            rect = (x1, y1, x2 - x1, y2 - y1)
            gc = GrabCut(img2, mask, rect)

    elif mode == "前景打点" :
        # 每次前景打点时更新掩码
        for prompt in new_points:
            if prompt in points:
                continue
            value = DRAW_FG
            x, y = int(prompt[0]), int(prompt[1])
            cv.circle(mask, (x, y), thickness, value, -1)

        gc.main()

    elif mode == "后景打点" :  # 确保 gc 已初始化
        # 每次后景打点时更新掩码
        for prompt in new_points:
            if prompt in points:
                continue
            value = DRAW_BG
            x, y = int(prompt[0]), int(prompt[1])
            cv.circle(mask, (x, y), thickness, value, -1)

        gc.main()

    state = {"img": img2, "mask": mask, "gc": gc, "rect": rect,"points":new_points}
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    output = cv.bitwise_and(img, img, mask=mask2)
    return output, state

demo = gr.Interface(
    fn=process_input,
    inputs=[
        ImagePrompter(show_label=False),
        gr.Radio(["矩形选框", "前景打点", "后景打点"], label="选择模式"),
        gr.Slider(2, 20, value=4, label="选择画笔粗细"),
        gr.State()
    ],
    outputs=[
        gr.Image(show_label=False),
        gr.State()
    ],
)

demo.launch()
