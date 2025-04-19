import os
import argparse
import numpy as np
import glob

def combine_upper_lower(upper_csv_path, lower_csv_path, output_path):
    """
    将上半身动作与下半身动作结合并保存到新文件
    """
    # 加载数据
    upper_data = np.genfromtxt(upper_csv_path, delimiter=',')
    lower_data = np.genfromtxt(lower_csv_path, delimiter=',')
    
    # 创建修改后的数据副本
    modified_data = lower_data.copy()
    
    # 确定上半身数据的开始索引
    upper_body_start_idx = 12 + 7
    
    # 确保两个数据集的长度匹配，取最小长度
    min_frames = min(upper_data.shape[0], lower_data.shape[0])
    modified_data = modified_data[:min_frames]
    
    # 将upper数据的上半身部分复制到modified_data中
    modified_data[:, upper_body_start_idx:] = upper_data[:min_frames, upper_body_start_idx:]
    
    # 重置腰部旋转
    modified_data[:,13+7] = 0
    modified_data[:,14+7] = 0
    
    # 保存混合后的数据
    np.savetxt(output_path, modified_data, delimiter=',')
    print(f"混合数据已保存至: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='批量混合上下半身动作')
    parser.add_argument('--upper_dir', type=str, required=True, help='包含上半身动作CSV文件的目录')
    parser.add_argument('--lower_file', type=str, required=True, help='下半身动作CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--robot_type', type=str, default='g1', help='机器人类型')
    
    args = parser.parse_args()
    
    # 确保路径格式正确
    upper_dir = os.path.normpath(args.upper_dir)
    lower_file = os.path.normpath(args.lower_file)
    output_dir = os.path.normpath(args.output_dir)
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取所有CSV文件
    upper_files = glob.glob(os.path.join(upper_dir, "*.csv"))
    
    if not upper_files:
        print(f"错误: 在 {upper_dir} 中未找到CSV文件")
        return
    
    # 检查下半身文件是否存在
    if not os.path.isfile(lower_file):
        print(f"错误: 下半身文件 {lower_file} 不存在")
        return
    
    print(f"找到 {len(upper_files)} 个CSV文件作为上半身动作")
    print(f"使用 {lower_file} 作为下半身动作")
    
    # 处理每个上半身文件
    combined_files = []
    for upper_file in upper_files:
        filename = os.path.basename(upper_file)
        output_filename = f"combined_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"处理: {filename}")
        combined_path = combine_upper_lower(upper_file, lower_file, output_path)
        combined_files.append(combined_path)
    
    print(f"完成! 共处理 {len(combined_files)} 个文件")

if __name__ == "__main__":
    main()