import os
import cv2

def add_border(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 如果图像为None，表示读取失败
    if image is None:
        print(f"无法读取图像：{image_path}")
        return
    
    # 检查图像尺寸是否为160x160
    if image.shape[:2] != (160, 160):
        print(f"图像尺寸不是160x160：{image_path}")
        return
    
    # 添加黑色边框
    bordered_image = cv2.rectangle(image, (0, 0), (159, 159), (0, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite(output_path, bordered_image)
    print(f"已保存图像：{output_path}")

def process_images_in_folder(folder_path):
    # 确保输出目录存在
    output_folder = os.path.join(folder_path, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # 构造输入和输出文件路径
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 添加边框并保存图像
            add_border(input_path, output_path)

if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "/home/dayang/second_paper/collect_code/drlnav_env-main/envs/map/wenhao_data/Map"  # 修改为你的文件夹路径
    
    # 处理文件夹下的所有图像
    process_images_in_folder(folder_path)
