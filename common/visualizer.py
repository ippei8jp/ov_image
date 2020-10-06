import os
import sys
import time
import math
import logging as log

import cv2
import numpy as np

# マウスイベントのコールバック
def onMouse(event, x, y, flag, params):
    visualizer = params[0]
    if event == cv2.EVENT_MOUSEWHEEL:
        # print(f'%%% WHEEL: 0x{flag:x}   ZOOM: {visualizer.ZOOM_TABLE[visualizer.zoom_index]}({visualizer.zoom_index})')
        if flag > 0 :
            # ZOOM IN
            visualizer.setWindowSize(1)
        elif flag < 0 :
            # ZOOM OUT
            visualizer.setWindowSize(-1)

class Visualizer:
    # 終了キー関連
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}
    
    # 最大/最小ウィンドウサイズ
    WINDOW_WIDTH_MAX  = 1700
    WINDOW_HEIGHT_MAX = 1000
    WINDOW_WIDTH_MIN  = 80
    WINDOW_HEIGHT_MIN = 60
    
    # ステータス表示部用
    STATUS_LINE_HIGHT    = 15                           # ステータス行の1行あたりの高さ
    STATUS_AREA_HIGHT    = STATUS_LINE_HIGHT * 6 + 8    # ステータス領域の高さは6行分と余白
    
    # 表示ズーム倍率テーブル
    ZOOM_TABLE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.50, 2.0, 4.0, 8.0)
    
    def __init__(self, args):
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_count = -1
        
        self.crop_size = args.crop
        
        self.frame_timeout = 0 if args.timelapse else 1
        
        # 表示ウィンドウ関連
        self.status_frame = None                            # ステータスフレーム
        self.enable_status_frame = not args.disable_status
        self.window_name = args.input
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, onMouse, [self])
        
        # 入出力ストリーム
        self.input_stream = None
        self.output_stream = None
        
        # open input file
        self.open_input_stream(args.input)
        
        # output stream
        self.open_output_stream(args.output)
    
    # 終了処理
    def terminete(self) :
        cv2.destroyAllWindows()
        if self.input_stream:
            self.input_stream.release()
        if self.output_stream:
            self.output_stream.release()
    
    # 入力ファイルのオープン
    def open_input_stream(self, path):
        log.info(f"Reading input data from {path}")
        p = path
        try:
            # pathは数字(カメラ指定)？
            p = int(path)
        except ValueError:
            # 数字でなければ絶対パスに変換
            p = os.path.abspath(path)
        
        # ファイル/カメラをオープン
        self.input_stream = cv2.VideoCapture(p)
        
        if self.input_stream is None or not self.input_stream.isOpened():
            # error
            log.error(f"Cannot open input stream '{path}'")
            raise FileNotFoundError(f"Cannot open input stream '{path}'")
        
        # get fps/frame size/total frames
        self.fps         =      self.input_stream.get(cv2.CAP_PROP_FPS)
        self.frame_count =  int(self.input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size  = (int(self.input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self.input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # cropping options
        if self.crop_size is not None:
            self.frame_size = tuple(np.minimum(self.frame_size, self.crop_size))
        
        log.info(f"Input stream info: {self.frame_size[0]} x {self.frame_size[1]} @ {self.fps:.2f} FPS")
        
        self.setWindowSize(0)
    
    # 入力ファイルがオープンされれているかチェック
    def check_input_stream(self) :
        if self.input_stream :
            return self.input_stream.isOpened()
        else :
            # オープンされていない
            return False
    
    # 入力フレーム取得
    def read_input_stream(self) :
        if self.input_stream :
            return self.input_stream.read()
        else :
            # オープンされていない
            return False, None
    
    # 出力ファイルのオープン
    def open_output_stream(self, path):
        if path :
            forcc = cv2.VideoWriter.fourcc(*'mp4v') if path.endswith('.mp4') else cv2.VideoWriter.fourcc(*'MJPG')
            log.info(f"Writing output to '{path}'")
            if self.enable_status_frame :
                self.output_stream = cv2.VideoWriter(path, forcc, self.fps, (self.frame_size[0], self.frame_size[1] + self.STATUS_AREA_HIGHT))
            else :
                self.output_stream = cv2.VideoWriter(path, forcc, self.fps, self.frame_size)
    
    # 出力ファイル書き込み
    def write_output_stream(self, frame):
        if self.output_stream :
           self.output_stream.write(frame)
    
    # ウィンドウサイズの変更
    def setWindowSize(self, param) :
        if param == 0 :
            window_width  = self.frame_size[0]          # とりあえず入力サイズで設定
            if self.enable_status_frame :
                window_height = self.frame_size[1] + self.STATUS_AREA_HIGHT
            else :
                window_height = self.frame_size[1]
            window_ratio  = window_width / window_height
            if   window_width >  self.WINDOW_WIDTH_MAX :    # 最大表示可能サイズを超える
                 window_width  =  self.WINDOW_WIDTH_MAX
                 window_height =  window_width / window_ratio
            elif window_width <  self.WINDOW_WIDTH_MIN :    # 最小表示可能サイズより小さい
                 window_width =  self.WINDOW_WIDTH_MIN
                 window_height =  window_width / window_ratio
             
            if   window_height > self.WINDOW_HEIGHT_MAX :   # 最大表示可能サイズを超える
                 window_height = self.WINDOW_HEIGHT_MAX
                 window_width  = window_height * window_ratio
            elif window_height < self.WINDOW_HEIGHT_MIN :   # 最小表示可能サイズより小さい
                 window_height = self.WINDOW_HEIGHT_MIN
                 window_width  = window_height * window_ratio
             
            self.disp_size_org = [int(window_width), int(window_height)]
            zoom_index = self.ZOOM_TABLE.index(1.0)
            disp_size = self.disp_size_org
        elif param > 0 :
            # ZOOM IN
            zoom_index = self.zoom_index + 1
            if zoom_index >= len(self.ZOOM_TABLE) :
                # 最大ZOOMに達している
                # print("**ZOOM ALREADY MAX**")
                return
            zoom_level =  self.ZOOM_TABLE[zoom_index]
            disp_size = [int(self.disp_size_org[0] * zoom_level), int(self.disp_size_org[1] * zoom_level)]
            if disp_size > [self.WINDOW_WIDTH_MAX, self.WINDOW_HEIGHT_MAX] :
                # 最大表示可能サイズを超える
                # print("**Larger than the displayable size**")
                return
        else :
            # ZOOM OUT
            zoom_index = self.zoom_index - 1
            if zoom_index < 0 :
                # 最小ZOOMに達している
                # print("**ZOOM ALREADY MIN**")
                return
            zoom_level =  self.ZOOM_TABLE[zoom_index]
            disp_size = [int(self.disp_size_org[0] * zoom_level), int(self.disp_size_org[1] * zoom_level)]
            if disp_size < [self.WINDOW_WIDTH_MIN, self.WINDOW_HEIGHT_MIN] :
                # 最小表示可能サイズより小さい
                # print("**Smaller than the displayable size**")
                return
        
        self.zoom_index = zoom_index
        # width と hightはタイトルバーを含むので、厳密にはこの値はずれている
        cv2.resizeWindow(self.window_name, disp_size[0], disp_size[1])
        # print(f'ZOOM index: {zoom_index} width: {disp_size[0]}   Height: {disp_size[1]}')
    
    # 中心部切り取り
    def center_crop(self, frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]
    
    # 入力の切り取り処理
    def crop_frame(self, frame) :
        if self.crop_size is not None:
            frame = self.center_crop(frame, self.crop_size)
        return frame
    
    # フレーム時間用タイマの更新
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
    
    # 文字列描画
    def draw_text(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(255, 255, 255), thickness=1):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline
    
    # 背景付き文字列描画
    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline
    
    # ROI枠描画
    def draw_detection_roi(self, frame, roi, label, bgcolor, txtcolor=(0,0,0)):
        cv2.rectangle(frame, tuple(roi.position), tuple(roi.position + roi.size), bgcolor, 2)
        
        # ラベル描画
        if label :
            text_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize("H1", font, text_scale, 1)
            line_height = np.array([0, text_size[0][1]])
            self.draw_text_with_background(frame, label,
                                           roi.position - line_height * 0.5,
                                           font, scale=text_scale,
                                           color=txtcolor, bgcolor=bgcolor)
    
    # ステータスフレームの生成と内容作成
    def draw_status(self, frame, detections, frame_number, inf_time):
        status_frame = np.zeros((self.STATUS_AREA_HIGHT, self.frame_size[0], 3), np.uint8)
        
        disp_frame_number = frame_number + 1
        origin = np.array([4, 18])              # 表示開始位置
        if disp_frame_number == self.frame_count :
            txtcolor = (128, 128, 255)
        else :
            txtcolor = (255, 255, 255)
        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_COMPLEX
        text_scale = 0.5
        frame_number_message    = f'frame_number   : {disp_frame_number:5d} / {self.frame_count}'
        frame_time_message      = f'Frame time     : {(self.frame_time * 1000):.3f} ms    {self.fps:.2f} fps'
        inf_time_message        = f'Inference time : {(inf_time * 1000):.3f} ms'
        text_size, _ = self.draw_text(status_frame, frame_number_message, origin, font, text_scale, txtcolor)
        origin = (origin + (0, text_size[1] * 1.5))
        text_size, _ = self.draw_text(status_frame, frame_time_message,   origin, font, text_scale, txtcolor)
        origin = (origin + (0, text_size[1] * 1.5))
        text_size, _ = self.draw_text(status_frame, inf_time_message,     origin, font, text_scale, txtcolor)
        
        log.debug(f'Frame: {disp_frame_number}/{self.frame_count}, ' \
                  f'frame time: {(self.frame_time * 1000):.3f}msec({self.fps:.2f}fps), ' \
                  f'Inference times: {(inf_time * 1000):.3f}msec')
        
        return status_frame
    
    # 画像フレームとステータスフレームを連結
    def make_display_frame(self, frame, status_frame=None) :
        if not self.enable_status_frame :
            return frame
        elif status_frame is None:
            return frame
        else :
            # 画像フレームとステータスフレームを連結
            return cv2.vconcat([frame, status_frame])
    
    # 表示
    def display_interactive_window(self, frame):
        """
        # 中断方法の表示
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = f"Press '{self.BREAK_KEY_LABELS}' key to exit"
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)
        """
        
        cv2.imshow(self.window_name, frame)
    
    # 表示の一時停止
    def should_stop_display(self, fevr=False) :
        if fevr :
            # wait forever
            key = cv2.waitKey(0) & 0xFF
        else :
            key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS
    
