@echo off
REM 构建脚本运行 Python rerun_visualize.py，处理available_walk_run.md中的所有clip

REM walk1_subject1 的各种步态
REM 慢走+转身
python rerun_visualize.py --file_name walk1_subject1 --robot_type g1 --clip True --start_frame 0 --end_frame 1385

REM 快直线
python rerun_visualize.py --file_name walk1_subject1 --robot_type g1 --clip True --start_frame 2480 --end_frame 2591

REM 快走圆弧转弯
python rerun_visualize.py --file_name walk1_subject1 --robot_type g1 --clip True --start_frame 2657 --end_frame 3117

REM 快走直角转弯
python rerun_visualize.py --file_name walk1_subject1 --robot_type g1 --clip True --start_frame 3163 --end_frame 3578

REM 小步慢走
python rerun_visualize.py --file_name walk1_subject1 --robot_type g1 --clip True --start_frame 5344 --end_frame 5772

REM 侧步慢走
python rerun_visualize.py --file_name walk1_subject1 --robot_type g1 --clip True --start_frame 6549 --end_frame 7846

echo 所有walk和run的clip处理完成!