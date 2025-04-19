import csv
import json
import math
import os
import re
import numpy as np

###############################################################################
# 1. Helper Functions
###############################################################################

def angle_axis_to_quaternion(angle_rad, axis):
    """
    Convert a single-axis rotation (angle in radians) about 'x', 'y', or 'z'
    into a quaternion [qx, qy, qz, qw].
    """
    half = angle_rad / 2.0
    s = math.sin(half)
    c = math.cos(half)
    
    if axis == 'x': # pitch
        return [0.0, s, 0.0, c]
    elif axis == 'y': # roll
        return [s, 0.0, 0.0, c]
    elif axis == 'z': # yaw
        return [0.0, 0.0, s, c]
    else:
        raise ValueError(f"Unknown axis '{axis}'. Must be 'x', 'y', or 'z'.")

def normalize_quaternion(q):
    """
    Normalize a quaternion to ensure it is a unit quaternion.
    
    Parameters:
    - q: List or tuple of quaternion components [qx, qy, qz, qw]
    
    Returns:
    - Normalized quaternion as a list [qx, qy, qz, qw]
    """
    norm = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if norm == 0:
        raise ValueError("Zero-norm quaternion is invalid.")
    return [component / norm for component in q]
###############################################################################
# 2. Joint Names and Axis Mapping
###############################################################################

# Complete list of all joint names in the CSV, in order (rows 7-35)
ALL_JOINT_NAMES = [
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link"
]

def infer_axis_from_joint_name(joint_name):
    """
    Infer the rotation axis based on the joint name.
    Adjust this function based on your specific conventions.
    """
    name_lower = joint_name.lower()
    
    if "pitch" in name_lower or "knee" in name_lower or "elbow" in name_lower:
        return 'x'
    elif "roll" in name_lower:
        return 'y'
    elif "yaw" in name_lower:
        return 'z'
    else:
        raise ValueError(f"Cannot infer axis for joint '{joint_name}'.")

###############################################################################
# 3. Main Conversion Function
###############################################################################

def convert_csv_to_json(
    csv_path,
    output_json_path,
    selected_joint_names,
    start_frame=0,
    end_frame=None,
    fps=30
):
    """
    Converts a CSV file to a JSON file, including only the selected joints
    and only the frames from start_frame to end_frame (inclusive).

    Parameters:
    - csv_path           : Path to the input CSV file.
    - output_json_path   : Path where the output JSON will be saved.
    - selected_joint_names : List of joint names to include in the JSON.
    - start_frame        : The first frame to include (0-based index).
    - end_frame          : The last frame to include (0-based index). If None, 
                           will include up to the last available frame.
    - fps                : Frames per second for the JSON data.

    JSON Structure:
    {
      "fps": <fps>,
      "frames": [
        {
          "pelvis": [[x, y, z], [qx, qy, qz, qw]],
          "selected_joint_1": [qx, qy, qz, qw],
          "selected_joint_2": [qx, qy, qz, qw],
          ...
        },
        ...
      ]
    }
    """
    # Step A: Read the raw CSV
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        all_rows = [list(map(float, row)) for row in reader]
    
    # all_rows is now a list of lists. Let's determine the shape.
    num_csv_rows = len(all_rows)           # Number of rows in CSV
    num_csv_cols = len(all_rows[0]) if num_csv_rows > 0 else 0  # Columns in first row
    print(num_csv_rows, num_csv_cols)
    
    # Verify final shape is [36, frames]
    if num_csv_cols != 36:
        raise ValueError(f"After transpose, expected 36 rows, but found {num_csv_cols} rows.")
    
    # print(f"Final CSV shape: [{num_csv_rows}, {num_csv_cols}] -> 36 DOFs x {num_csv_cols} frames.")
    
    # Step C: Confirm each row has the same number of columns
    for idx, row in enumerate(all_rows):
        if len(row) != num_csv_cols:
            raise ValueError(f"Row {idx} has {len(row)} columns; expected {num_csv_cols}.")

    # Step D: Validate selected joints
    invalid_joints = [joint for joint in selected_joint_names if joint not in ALL_JOINT_NAMES]
    if invalid_joints:
        raise ValueError(f"The following selected joints are not recognized: {invalid_joints}")
    
    # print(f"Number of selected joints: {len(selected_joint_names)}")
    
    # Step E: Map selected joints to their row indices
    # Rows 7 to 35 correspond to ALL_JOINT_NAMES[0..28]
    joint_to_row_idx = {joint: 7 + idx for idx, joint in enumerate(ALL_JOINT_NAMES)}
    
    # Retrieve the row indices for the selected joints
    selected_joint_row_indices = {joint: joint_to_row_idx[joint] for joint in selected_joint_names}
    
    # Step F: Determine valid frame range
    total_frames = num_csv_rows
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(
            f"start_frame ({start_frame}) is out of range [0..{total_frames-1}]."
        )
    if end_frame is None or end_frame >= total_frames:
        end_frame = total_frames - 1  # Include up to last frame
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame ({end_frame}) must be >= start_frame ({start_frame})."
        )
    
    # print(f"Selecting frames from {start_frame} to {end_frame} (inclusive).")
    
    # Step G: Iterate over each frame in the specified range
    frames_out = []
    for frame_idx in range(start_frame, end_frame + 1):
        frame_dict = {}
        
        # Extract root joint data (rows 0-6)
        x  = all_rows[frame_idx][0]
        y  = all_rows[frame_idx][1]
        z  = all_rows[frame_idx][2] # + 0.03
        qx = all_rows[frame_idx][3]
        qy = all_rows[frame_idx][4]
        qz = all_rows[frame_idx][5]
        qw = all_rows[frame_idx][6]
        
        # Normalize the root quaternion
        # qx, qy, qz, qw = quaternion_xyz_to_zyx(qx, qy, qz, qw)
        frame_dict["pelvis"] = [
            [x, y, z],       # Position
            normalize_quaternion([qx, qy, qz, qw]) # Orientation (normalized)
        ]
        # print(x, y, z, qx, qy, qz)
        
        # Process each selected joint
        for joint_name in selected_joint_names:
            row_idx = selected_joint_row_indices[joint_name]
            angle_deg = all_rows[frame_idx][row_idx]
            
            # Convert degrees to radians (if applicable)
            # If your data is already in radians, comment out the next line
            # angle_rad = math.radians(angle_deg)
            angle_rad = angle_deg
            
            # Infer rotation axis
            axis = infer_axis_from_joint_name(joint_name)
            
            # Convert angle to quaternion
            q = angle_axis_to_quaternion(angle_rad, axis)
            
            # Normalize quaternion to ensure it's a unit quaternion
            q = normalize_quaternion(q)
            
            # idx_mapping = {
            #     'x': 0, 'y': 1, 'z': 2
            # }
            
            # if 'roll' in joint_name:
            #     ang = quat2expmap(torch.tensor(q))
            #     print("angle_rad", angle_rad)
            #     print("ang_reverse {:.6f}".format(ang[idx_mapping[axis]])) 
            #     print("axis", axis)
            #     print("q", q)
            #     assert 0
            
            # Assign to frame dictionary
            frame_dict[joint_name] = q
        
        frames_out.append(frame_dict)
        
        # Optional: Print progress every 1000 frames
        # if (frame_idx - start_frame + 1) % 1000 == 0:
        #     print(f"Processed {frame_idx - start_frame + 1} frames in selection...")
    
    # Step H: Assemble the final JSON structure
    output_data = {
        "fps": fps,
        "frames": frames_out
    }
    
    # Step I: Write to JSON file
    with open(output_json_path, 'w') as out_f:
        json.dump(output_data, out_f, indent=2)
    
    print(
        f"Successfully wrote JSON with {len(frames_out)} frames (from "
        f"frame {start_frame} to {end_frame}) to '{output_json_path}'."
    )

import yaml
import os

def generate_yaml(files, output_file):
    """
    Generate a YAML file with the following format:

    motions:
    - file: generated/{filename}
      weight: null

    Parameters:
        files (list): List of file names.
        output_file (str): Path to the output YAML file.
    """
    # Build the YAML data structure
    data = {
        "motions": [
            {"file": f"generated/{file}", "weight": None} for file in files
        ]
    }

    # Write the data to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
###############################################################################
# 4. Example Usage
###############################################################################

if __name__ == "__main__":
    # Example of a reduced set of selected joints
    SELECTED_JOINT_NAMES = [
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "waist_pitch_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link"
    ]
    
    # 
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = r"E:\CS\Graphics\lafan-g1"
    
    csv_directory = os.path.join(current_dir, "g1", "combined_clips")
    json_directory = os.path.join(current_dir, "json", "good_combined_clips")
    if not os.path.exists(json_directory):
        os.makedirs(json_directory)
    csv_files = os.listdir(csv_directory)
    json_files = [csv_file.replace(".csv", ".json") for csv_file in csv_files]
    all_motions = [{"file": file} for file in json_files]
    clip_frames = False
    generate_yaml(files=json_files, output_file=os.path.join(current_dir, "all.yaml"))
    
    for motion in all_motions:
        json_file_path = motion["file"]
        
        output_file_path = os.path.join(json_directory, json_file_path)
        filename = os.path.basename(json_file_path)
        if clip_frames:
            match = re.match(r"(.+)_(\d+)_(\d+)\.json", filename)
            if not match:
                print(f"Format is not correct: {filename}")
                continue
            
            base_name = match.group(1)  # e.g., lafan1_walk1_subject1
            start_frame = int(match.group(2))  # e.g., 100
            end_frame = int(match.group(3))    # e.g., 1170
            print(base_name, start_frame, end_frame)
        
            csv_file_name = f"{base_name}.csv"
            csv_file_path = os.path.join(csv_directory, csv_file_name)
        else:
            start_frame = 0
            end_frame = None
            csv_file_name = motion['file'].replace('.json', '.csv')
            csv_file_path = os.path.join(csv_directory, csv_file_name)
        
        if not os.path.isfile(csv_file_path):
            print(f"Can not find: {csv_file_path}, skip it")
            continue
        
        try:
            convert_csv_to_json(
                csv_path=csv_file_path,
                output_json_path=output_file_path,
                selected_joint_names=SELECTED_JOINT_NAMES,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=30
            )
        except Exception as e:
            print(f"Processing {output_file_path} Error: {e}")


    