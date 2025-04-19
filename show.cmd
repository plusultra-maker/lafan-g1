@echo off
REM 构建脚本运行 Python rerun_visualize.py

REM python rerun_visualize.py --upper_file_name fightAndSports1_subject4129857 --robot_type g1 --use_connect True 

python rerun_visualize.py --file_name walk1_subject1_6549_7846 --robot_type g1

REM 构建脚本运行 Python rerun_visualize.py，处理available.md中的所有clip

REM fight1_subject2 - 没有特定片段

REM REM fight1_subject3
REM python rerun_visualize.py --file_name fight1_subject3 --robot_type g1 --clip True --start_frame 1075 --end_frame 1326
REM python rerun_visualize.py --file_name fight1_subject3 --robot_type g1 --clip True --start_frame 6743 --end_frame 6824
REM 
REM REM fight1_subject5
REM python rerun_visualize.py --file_name fight1_subject5 --robot_type g1 --clip True --start_frame 5410 --end_frame 5497
REM python rerun_visualize.py --file_name fight1_subject5 --robot_type g1 --clip True --start_frame 5910 --end_frame 6080
REM python rerun_visualize.py --file_name fight1_subject5 --robot_type g1 --clip True --start_frame 6964 --end_frame 7161
REM 
REM REM fightAndSports1_subject1
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 2740 --end_frame 2875
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 2880 --end_frame 3142
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 4118 --end_frame 4284
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 4557 --end_frame 4724
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 4808 --end_frame 5036
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 5456 --end_frame 5832
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 6256 --end_frame 6522
REM python rerun_visualize.py --file_name fightAndSports1_subject1 --robot_type g1 --clip True --start_frame 6634 --end_frame 7354
REM 
REM REM fightAndSports1_subject4 (含best标记)
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 153 --end_frame 809
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 1835 --end_frame 2162
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 2476 --end_frame 2596
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 3082 --end_frame 3192
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 3586 --end_frame 3768
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 4378 --end_frame 4553
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 4765 --end_frame 4950
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 5856 --end_frame 6032
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 6697 --end_frame 6787
REM python rerun_visualize.py --file_name fightAndSports1_subject4 --robot_type g1 --clip True --start_frame 7139 --end_frame 7354
REM 
REM echo 所有clip处理完成!