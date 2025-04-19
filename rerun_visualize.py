import argparse
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
import os

class RerunURDF():
    def __init__(self, robot_type):
        self.name = robot_type
        match robot_type:
            case 'g1':
                self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/g1/g1_29dof_rev_1_0.urdf', 'robot_description/g1', pin.JointModelFreeFlyer())
                self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                       -0.15,0,0,0.3,-0.15,0,
                                       -0.15,0,0,0.3,-0.15,0,
                                       0,0,0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,-1.57,0,1.57,0,0,0]).astype(np.float32)
            case 'h1_2':
                self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/h1_2/h1_2_wo_hand.urdf', 'robot_description/h1_2', pin.JointModelFreeFlyer())
                assert self.robot.model.nq == 7 + 12+1+14
                self.Tpose = np.array([0,0,1.02,0,0,0,1,
                                       0,-0.15,0,0.3,-0.15,0,
                                       0,-0.15,0,0.3,-0.15,0,
                                       0,
                                       0, 1.57,0,1.57,0,0,0,
                                       0,-1.57,0,1.57,0,0,0]).astype(np.float32)
            case 'h1':
                self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/h1/h1.urdf', 'robot_description/h1', pin.JointModelFreeFlyer())
                assert self.robot.model.nq == 7 + 10+1+8
                self.Tpose = np.array([0,0,1.03,0,0,0,1,
                                       0,0,-0.15,0.3,-0.15,
                                       0,0,-0.15,0.3,-0.15,
                                       0,
                                       0, 1.57,0,1.57,
                                       0,-1.57,0,1.57]).astype(np.float32)
            case _:
                print(robot_type)
                raise ValueError('Invalid robot type')
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject2')
    parser.add_argument('--clip', type=bool, help="Clip", default=False)
    parser.add_argument('--use_connect', type=bool, help="Use connect", default=False)
        # 在rerun_visualize.py的参数部分添加
    parser.add_argument('--start_frame', type=int, help="Start frame for clipping", default=0)
    parser.add_argument('--end_frame', type=int, help="End frame for clipping", default=None)
    parser.add_argument('--upper_file_name', type=str, help="Upper file name", default='dance1_subject2')
    parser.add_argument('--lower_file_name', type=str, help="Lower file name", default='lift_feet_walk1_subject1')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    args = parser.parse_args()

    rr.init('Reviz', spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    file_name = args.file_name
    robot_type = args.robot_type
    csv_files = robot_type + '/' + file_name + '.csv'
    if args.use_connect:    
        upper_file_name = args.upper_file_name
        lower_file_name = args.lower_file_name
        upper_csv_files = robot_type + '/' + upper_file_name + '.csv'
        lower_csv_files = robot_type + '/' + lower_file_name + '.csv'
        upper_data = np.genfromtxt(upper_csv_files, delimiter=',')
        lower_data = np.genfromtxt(lower_csv_files, delimiter=',')
    else:
        data = np.genfromtxt(csv_files, delimiter=',')
        modified_data = data.copy()
        waist_x_rot = modified_data[:,13+7].copy()
        modified_data[:,13+7] = 0
        modified_data[:,14+7] = 0
        
        # 调整base rotation
        base_rot = modified_data[:,3:7]
        for i in range(modified_data.shape[0]):
            # 创建绕X轴旋转的四元数 [qx, qy, qz, qw]
            angle = waist_x_rot[i] / 2.0
            qx = np.sin(angle)
            qy = 0.0
            qz = 0.0
            qw = np.cos(angle)
            x_rot_quat = np.array([qx, qy, qz, qw])
            
            # 应用四元数乘法: base_rot = x_rot_quat * base_rot
            # 四元数乘法的实现
            q1 = x_rot_quat
            q2 = base_rot[i]
            
            result_w = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]
            result_x = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1]
            result_y = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0]
            result_z = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3]
            
            # 归一化四元数
            mag = np.sqrt(result_w**2 + result_x**2 + result_y**2 + result_z**2)
            if mag > 0:
                base_rot[i] = np.array([result_x, result_y, result_z, result_w]) / mag

        # 将结果存回modified_data
        modified_data[:,3:7] = base_rot
        modified_data[:,1+7] -= waist_x_rot
        modified_data[:,7+7] -= waist_x_rot
        
    if args.use_connect:
        modified_data = lower_data.copy()
        upper_body_start_idx = 12 + 7

        # 确保两个数据集的长度匹配，取最小长度
        print(f"upper_data length: {upper_data.shape[0]}")
        print(f"lower_data length: {lower_data.shape[0]}")
        
        upper_start_idx = 0
        upper_stop_idx = 337
        upper_selected_length = upper_stop_idx - upper_start_idx
        upper_selected_data = upper_data[upper_start_idx:upper_stop_idx, :]
        print(f"upper_selected_data length: {upper_selected_data.shape[0]}")
        lower_start_idx = 0
        lower_stop_idx = lower_data.shape[0]
        lower_length = lower_stop_idx - lower_start_idx
        print(f"lower_data length: {lower_length}")
        min_frames = min(upper_selected_length, lower_length)
        
        # 将upper数据的上半身部分复制到modified_data中
        # 上半身数据从索引19开始到结束
        if lower_length >= min_frames:
            modified_data[lower_start_idx:lower_start_idx+upper_selected_length, upper_body_start_idx:] = upper_data[upper_start_idx:upper_stop_idx, upper_body_start_idx:]
            modified_data = modified_data[lower_start_idx:lower_start_idx+upper_selected_length, :]
        else:
            modified_data[lower_start_idx:lower_start_idx+min_frames, upper_body_start_idx:] = upper_data[upper_start_idx:upper_start_idx+min_frames, upper_body_start_idx:]
        # 保存混合后的数据
        
        modified_data[:,13+7] = 0
        modified_data[:,14+7] = 0
        output_filename = f"{robot_type}/combined_{upper_file_name}_{lower_file_name}.csv"
        np.savetxt(output_filename, modified_data, delimiter=',')
        print(f"混合数据已保存至: {output_filename}")
    # stand to walk
    # 1-30 站立 悬垂手臂
    #for i in range(30):
    #    modified_data[i, 16+7] -= 1.3
    #    modified_data[i, 23+7] += 1.3
    ## 30-80 慢慢举起双臂
    #for i in range(30, 80):
    #    offset = 1.3*(1-(i-30)/50.)
    #    modified_data[i, 16+7] -= offset
    #    modified_data[i, 23+7] += offset
    # 其余 慢慢放下双臂
    # 先只截取0-f2
    modify_walk = False
    if modify_walk:
        f1=134
        f2=178
        modified_data[:100, :] = modified_data[100, :]
        #modified_data = modified_data[:f2, :]
        modified_data = modified_data[:460, :]
        modified_data[360:, :] = modified_data[360, :]
        modified_data[:,16+7] += 0.3
        modified_data[:,23+7] -= 0.3
        waist_x_rot = modified_data[:,13+7].copy()
        modified_data[:,13+7] = 0
        modified_data[:,14+7] = 0
        
        # 调整base rotation
        base_rot = modified_data[:,3:7]
        for i in range(modified_data.shape[0]):
            # 创建绕X轴旋转的四元数 [qx, qy, qz, qw]
            angle = waist_x_rot[i] / 2.0
            qx = np.sin(angle)
            qy = 0.0
            qz = 0.0
            qw = np.cos(angle)
            x_rot_quat = np.array([qx, qy, qz, qw])
            
            # 应用四元数乘法: base_rot = x_rot_quat * base_rot
            # 四元数乘法的实现
            q1 = x_rot_quat
            q2 = base_rot[i]
            
            result_w = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]
            result_x = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1]
            result_y = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0]
            result_z = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3]
            
            # 归一化四元数
            mag = np.sqrt(result_w**2 + result_x**2 + result_y**2 + result_z**2)
            if mag > 0:
                base_rot[i] = np.array([result_x, result_y, result_z, result_w]) / mag

        # 将结果存回modified_data
        modified_data[:,3:7] = base_rot
        modified_data[:,1+7] -= waist_x_rot
        modified_data[:,7+7] -= waist_x_rot
        # 对136-181做100次loop，并在交接处blend
        
        #one_cycle = modified_data[f1:f2, :].copy()
        #cycle_length = one_cycle.shape[0]  
        #blend_frames = 10  # 用于平滑过渡的帧数
        #n = 3
        ## 创建足够大的数组存储所有循环
        #total_frames = f2 + cycle_length * n
        #extended_data = np.zeros((total_frames, modified_data.shape[1]))
        #extended_data[:f2, :] = modified_data  # 复制原始数据
        #
        ## 计算原始循环首尾的XY位移差值
        #xy_stride = (modified_data[f1, 0:2] - modified_data[f1-1, 0:2])*1.1
        #xy_offset = 0
        #for i in range(n):
        #    start_idx = f2 + i * cycle_length
        #    end_idx = start_idx + cycle_length
        #    
        #    # 复制一个新的循环
        #    current_cycle = one_cycle.copy()
        #    
        #    # 计算当前循环的XY偏移量
        #    if i == 0:
        #        # 第一个循环，与原始数据的最后一帧对齐
        #        xy_offset = modified_data[f2-1, 0:2] - current_cycle[0, 0:2] + xy_stride
        #    else:
        #        # 后续循环，累加位移差值
        #        xy_offset = extended_data[start_idx-1, 0:2] - one_cycle[0, 0:2] + xy_stride
        #    
        #    # 应用XY偏移
        #    current_cycle[:, 0:2] += xy_offset
        #    
        #    # 处理Z坐标和旋转的平滑过渡
        #    if i >= 0:
        #        for j in range(blend_frames):
        #            if j >= cycle_length:
        #                break
        #                
        #            # 计算混合因子 (0->1)
        #            t = (j+1) / blend_frames
        #            blend_factor = t
        #            
        #            # 获取前一循环结束附近的帧
        #            prev_frame_idx = start_idx - 1
        #            if j == 0:
        #                # xy坐标平滑过渡
        #                #current_cycle[j, 0] = extended_data[prev_frame_idx, 0] * (1 - blend_factor) + current_cycle[j, 0] * blend_factor
        #                #current_cycle[j, 1] = extended_data[prev_frame_idx, 1] * (1 - blend_factor) + current_cycle[j, 1] * blend_factor
        #                # Z坐标平滑过渡
        #                current_cycle[j, 2] = extended_data[prev_frame_idx, 2] * (1 - blend_factor) + current_cycle[j, 2] * blend_factor
        #                
        #                # 四元数(根部方向)平滑过渡
        #                current_cycle[j, 3:7] = extended_data[prev_frame_idx, 3:7] * (1 - blend_factor) + current_cycle[j, 3:7] * blend_factor
        #                
        #                # 归一化四元数
        #                quat_norm = np.linalg.norm(current_cycle[j, 3:7])
        #                if quat_norm > 0:
        #                    current_cycle[j, 3:7] /= quat_norm
        #                
        #                # 对关节角度也进行平滑过渡
        #                current_cycle[j, 7:] = extended_data[prev_frame_idx, 7:] * (1 - blend_factor) + current_cycle[j, 7:] * blend_factor
        #            else:
        #                #current_cycle[j, 0] = current_cycle[j-1, 0] * (1 - blend_factor) + current_cycle[j, 0] * blend_factor
        #                #current_cycle[j, 1] = current_cycle[j-1, 1] * (1 - blend_factor) + current_cycle[j, 1] * blend_factor
        #                current_cycle[j, 2] = current_cycle[j-1, 2] * (1 - blend_factor) + current_cycle[j, 2] * blend_factor
        #                current_cycle[j, 3:7] = current_cycle[j-1, 3:7] * (1 - blend_factor) + current_cycle[j, 3:7] * blend_factor
        #                quat_norm = np.linalg.norm(current_cycle[j, 3:7])   
        #                if quat_norm > 0:
        #                    current_cycle[j, 3:7] /= quat_norm
        #                current_cycle[j, 7:] = current_cycle[j-1, 7:] * (1 - blend_factor) + current_cycle[j, 7:] * blend_factor
        #    
        #    # 将当前循环添加到扩展数据中
        #    extended_data[start_idx:end_idx, :] = current_cycle
        #
        ## 更新modified_data为扩展后的数据
        #modified_data = extended_data
        
        # 这个部分是要标注数据中那部分是左脚离地哪部分的右脚离地，然后分别处理
            # 这个部分是要标注数据中那部分是左脚离地哪部分的右脚离地，然后分别处理
        left_lift_clips = [[138,149],[181,194],[225,238],[270,283]]
        right_lift_clips = [[114,129],[159,170],[202,215],[246,259]]
        right_knee_idx = 9+7
        left_knee_idx = 3+7
        left_ankle_pitch_idx = 4+7
        right_ankle_pitch_idx = 10+7
        
        # 左脚平滑过渡
        for clip in left_lift_clips:
            start, end = clip
            duration = end - start
            for i in range(duration):
                # 使用正弦曲线创建平滑的过渡效果
                # sin函数在[0,π]范围内从0到1再回到0
                progress = i / duration
                if progress < 0.3:
                    # 前半段：从0增加到最大值
                    factor = progress
                elif progress < 0.7:
                    # 中间段：保持最大值
                    factor = 0.3
                else:
                    # 后半段：从最大值减少到0
                    factor = 1 - progress
                
                # 应用变化因子到膝盖和脚踝
                modified_data[start + i, left_knee_idx] += 1 * factor
                modified_data[start + i, left_ankle_pitch_idx] -= 1 * factor
        
        # 右脚平滑过渡
        for clip in right_lift_clips:
            start, end = clip
            duration = end - start
            for i in range(duration):
                progress = i / duration
                if progress < 0.3:
                    factor = progress
                elif progress < 0.7:
                    factor = 0.3
                else:
                    factor = 1 - progress
                
                # 对右脚应用类似的调整
                modified_data[start + i, right_knee_idx] += 1 * factor
                modified_data[start + i, right_ankle_pitch_idx] -= 1 * factor
    
    # simple clip
    # 替换rerun_visualize.py中的simple clip部分
    if not args.use_connect and args.clip:
        start = args.start_frame
        end = args.end_frame if args.end_frame is not None else modified_data.shape[0]
        modified_data = modified_data[start:end, :]
        start_str = str(start)
        end_str = str(end)
        # 创建目录（如果不存在）
        output_dir = robot_type + '/' + 'good_clips'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存
        output_file = os.path.join(output_dir, file_name + '_' + start_str + '_' + end_str + '.csv')
        np.savetxt(output_file, modified_data, delimiter=',')
        print(f"数据已保存至: {output_file}")
    data = modified_data
    rerun_urdf = RerunURDF(robot_type)
    ##for i in range (29):
    ##    print(i,rerun_urdf.robot.model.names[i])
    ##exit(0)
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = data[frame_nr, :]
        rerun_urdf.update(configuration)