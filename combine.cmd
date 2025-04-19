@echo off
REM 批量生成混合动作
python combine_motions.py --upper_dir "g1/good_clips" --lower_file "g1/lift_feet_walk1_subject1.csv" --output_dir "g1/combined_clips"
echo 所有混合动作已生成!