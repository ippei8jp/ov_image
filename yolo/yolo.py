#!/usr/bin/env python

import os
import sys
import time
import logging as log
import pprint
from argparse import ArgumentParser

# importのサーチパスを追加
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"../common"))

"""
# 環境変数設定スクリプトが実行されているか確認 =======================================
if not "INTEL_OPENVINO_DIR" in os.environ:
    print("**** ERROR !!!! ****")
    print("Script doesn't seem to be running. ")
    print("Please run `source /opt/intel/openvino/bin/setupvars.sh`")
    raise  OSError("openVINO environments is not set.")
else:
    # 環境変数を取得するには os.environ['INTEL_OPENVINO_DIR']
    # これを設定されてない変数に対して行うと例外を吐くので注意
    pass
# ====================================================================================
"""

# import cv2
# import numpy as np

# ユーザ定義モジュール
from yolo_visualizer import YoloVisualizer
from yolo_frame_processor import YoloFrameProcessor

# 使用可能なデバイスのリスト
DEVICE_KINDS = ['CPU', 'GPU', 'MYRIAD']

def build_argparser():
    parser = ArgumentParser()
    # input
    parser.add_argument('input', metavar="INPUT_FILE", 
                         help="Path to the input video/picture ")
    general = parser.add_argument_group('General')
    # output
    general.add_argument('-o', '--output', metavar="OUTPUT_FILE", default="",
                         help="(optional) Path to save the output video to")
    # cropping
    general.add_argument('--crop', default=None, type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                         help="(optional) Crop the input stream to this size (default: no crop).")
    # liblary
    general.add_argument('-l', '--cpu_lib', metavar="LIB_PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    general.add_argument('-c', '--gpu_lib', metavar="LIB_PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    # misc
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    general.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    general.add_argument('--disable_status', action='store_true',
                         help="(optional) Disable status frame")
    
    # for YOLO Detector
    sdetect = parser.add_argument_group('YOLO')
    sdetect.add_argument('-m_yolo', metavar="MODEL_PATH", default="", required=True,
                        help="Path to the Face Detection model XML file")
    sdetect.add_argument('-d_yolo', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    sdetect.add_argument('-t_yolo', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for YOLO" \
                       "(default: %(default)s)")
    sdetect.add_argument("--iou_threshold", default=0.4, type=float, \
                       help="(optional) Intersection over union threshold "
                       "for overlapping detections filtering" \
                       "(default: %(default)s)")
    return parser


def main() :
    # get options
    args = build_argparser().parse_args()
    
    # log setting
    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)
    log.debug(str(args))
    
    # Pre-process
    visualizer = YoloVisualizer(args)
    frame_processor = YoloFrameProcessor(args)
    
    # display flag
    display = not args.no_show
    
    # frame number
    frame_number = 0
    
    # break or normal end
    break_flag = False
    
    # for initialize frame timer
    visualizer.update_fps()
    
    # main loop
    while visualizer.check_input_stream() :
        # frame_start_time = time.time()
        # input frame
        has_frame, frame = visualizer.read_input_stream()
        if not has_frame:
            # end of frame
            break
        
        # cropping
        frame = visualizer.crop_frame(frame)
        
        # Recognition process
        inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        rois = frame_processor.process(frame)
        inf_end = time.time()                                   # 推論処理終了時刻          --------------------------------
        inf_time = inf_end - inf_start                          # 推論処理時間
        
        # Result output
        visualizer.draw_detections(frame, rois)
        visualizer.update_fps()
        status_frame = visualizer.draw_status(frame, rois, frame_number, inf_time)
        if args.perf_stats:
            log.info('Performance stats:')
            # log.info(pprint.pformat(frame_processor.get_performance_stats()))
            pprint.pprint(frame_processor.get_performance_stats())
        
        frame = visualizer.make_display_frame(frame, status_frame)
        
        # output to file
        visualizer.write_output_stream(frame)
        
        if display:
            visualizer.display_interactive_window(frame)
            break_flag = visualizer.should_stop_display()
            if break_flag :
                break
        
        frame_number += 1
        # frame_time = time.time()- frame_start_time
        # print(f'frame_time : {frame_time}')
    
    # Hold the window waiting for keystrokes at the last frame
    if display:                                             # display mode
        if not break_flag :                                 # not break loop
            if frame_number > 0 :                           # no input error
                print("Press any key to exit")
                visualizer.should_stop_display(True)
    
    # Release resources
    visualizer.terminete()

if __name__ == '__main__':
    main()
