import cv2
import numpy as np
from Grabcut import GrabCut

def run_grabcut(img, mask, iterations=5, rect=None):
    """
    封装的GrabCut算法执行函数。

    参数:
    img - 输入图像 (numpy.ndarray)
    mask - 输入和输出掩码 (numpy.ndarray)
    iterations - GrabCut算法迭代次数 默认为5
    rect - 初始化的矩形区域，格式为(x, y, width, height)

    返回:
    更新后的掩码
    """
    # GrabCut算法内部使用的模型参数，这里初始化为空的数组
    bgdModel = np.zeros((1, 65), np.float64)  # 背景模型
    fgdModel = np.zeros((1, 65), np.float64)  # 前景模型
    
    # 如果没有提供矩形区域
    if rect is None:
        rect = (50, 50, img.shape[1]-100, img.shape[0]-100)

    # 运行GrabCut算法
    # iterations 图割算法运行的次数影响分割的精确度和计算时间。
    # cv2.GC_INIT_WITH_RECT 使用提供的矩形作为前景的初始猜测来启动分割过程。
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    
    return mask  # 返回更新后的掩码

def select_roi(img):  
    """
    该函数允许用户通过交互方式在图像上选择一个感兴趣区域(ROI)。

    参数:
    img - 输入图像 (numpy.ndarray), 需要用户在其上选择ROI的图像。

    返回:
    rect - 用户选择的矩形区域坐标元组 (r, c, w, h)
           r为矩形顶部y坐标     c为矩形左侧x坐标
           w为矩形宽度          h为矩形高度。
    """
    # 创建一个名为 'Select ROI' 的窗口，设置窗口的大小为 500x500
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select ROI', 800, 500)

    r, c, w, h = cv2.selectROI('Select ROI', img, False, False)  
    rect = (r, c, w, h)
  
    # 等待用户按2下空格，然后关闭窗口  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
  
    return rect

def Grabcut(image_path):
    """
    使用GrabCut算法进行前景提取的完整演示函数 包括用户交互式选择ROI

    参数:
    image_path (str): 图像文件的路径。
    """

    # 加载图片
    img = cv2.imread(image_path)

    if img is None:
        print("Cannot load the image!")
        return
    else:
        height, width = img.shape[:2]
        print("Image width:", width)
        print("Image height:", height)

    
    # 初始化掩码，大小与原图相同，数据类型为无符号8位整型。
    # 0表示确定的背景，1表示确定的前景，2表示可能的背景，3表示可能的前景
    # img.shape[:2]获得长 宽
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = select_roi(img)

    # 调用封装的GrabCut函数
    # updated_mask = run_grabcut(img, mask, rect=rect)
    grabcut_model = GrabCut(img, mask, rect)
    grabcut_model.GMM_3D()
    updated_mask = grabcut_model.mask


    # 根据掩码创建一个新的图像，只显示确定的和可能的前景部分（掩码值为2或3的部分）
    # 将掩码值为2或0的位置设为0，表示这些像素不显示；其余（即确定或可能的前景）设为1
    # 最后转换为与原图相同的深度以进行乘法操作
    mask2 = np.where((updated_mask == 2) | (updated_mask == 0), 0, 1).astype('uint8')
    img_ = img * mask2[:, :, np.newaxis]  # 应用掩码到原图 增加一个空的轴（使用np.newaxis）

    # 创建一个名为 'GrabCut Foreground Extraction' 的窗口，设置窗口的大小为 500x500
    cv2.namedWindow('GrabCut Foreground Extraction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('GrabCut Foreground Extraction', 800, 500)

    # 显示处理后的图像，仅包含提取出的前景
    cv2.imshow("GrabCut Foreground Extraction", img_)
    while 1:
        # 等待用户输入
        k = cv2.waitKey(1)
        if k == ord('n'):
            grabcut_model.main()
            grabcut_model.GMM_3D()
        elif k == 27:
            break
    # 关闭所有OpenCV创建的窗口
    cv2.destroyAllWindows()

# 测试代码：指定图片路径并调用函数
image_path = '00.jpg' 
grabCut = Grabcut(image_path)