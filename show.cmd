@echo off
REM 构建脚本运行 Python rerun_visualize.py

REM python rerun_visualize.py --upper_file_name fightAndSports1_subject4129857 --robot_type g1 --use_connect True 

REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1

REM 构建脚本运行 Python rerun_visualize.py，处理available.md中的所有clip

REM fight1_subject2 - 没有特定片段

REM fight1_subject3
python rerun_visualize.py --file_name fight1_subject3 --robot_type g1 --clip True --start_frame 1075 --end_frame 1326
python rerun_visualize.py --file_name fight1_subject3 --robot_type g1 --clip True --start_frame 6743 --end_frame 6824

REM fight1_subject5
python rerun_visualize.py --file_name fight1_subject5 --robot_type g1 --clip True --start_frame 5410 --end_frame 5497
python rerun_visualize.py --file_name fight1_subject5 --robot_type g1 --clip True --start_frame 5910 --end_frame 6080
python rerun_visualize.py --file_name fight1_subject5 --robot_type g1 --clip True --start_frame 6964 --end_frame 7161

REM fightAndSports1_subject1
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 2740 --end_frame 2875
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 2880 --end_frame 3142
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 4118 --end_frame 4284
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 4557 --end_frame 4724
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 4808 --end_frame 5036
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 5456 --end_frame 5832
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 6256 --end_frame 6522
python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 6634 --end_frame 7354

REM fightAndSports1_subject4 (含best标记)
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 153 --end_frame 809
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 1835 --end_frame 2162
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 2476 --end_frame 2596
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 3082 --end_frame 3192
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 3586 --end_frame 3768
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 4378 --end_frame 4553
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 4765 --end_frame 4950
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 5856 --end_frame 6032
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 6697 --end_frame 6787
python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 7139 --end_frame 7354

echo 所有clip处理完成!