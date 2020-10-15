# SSD処理

## ファイル構成

| ファイル                | 内容                                             |
|-------------------------|--------------------------------------------------|
| demo.sh                 | スクリプト実行コマンド                           |
| _model_download.sh      | モデルファイルをダウンロードするスクリプト       |
|                         |                                                  |
| yolo.py                 | YOLO処理プログラム本体                           |
| yolo_frame_processor.py | フレーム毎の処理クラス                           |
| yolo_visualizer.py      | 画像関連管理クラス                               |
|                         |                                                  |
| yolo_detector.py        | YOLO検出器クラス                                 |


## YOLO処理プログラム実行方法

あらかじめ`` bash _model_download.sh``を使用してモデルファイルをダウンロードしてください。  
``models``ディレクトリにモデルファイルがダウンロード(またはダウンロード＆コンバート)されます。  
いくつかのモデルファイルをダウンロード(またはダウンロード＆コンバート)しますので、
不要なモデルがある場合は、それをダウンロードする部分(``download_XXXX_XXXX``の呼び出し部分)を
コメントアウトしてください。  

yolo.pyが処理本体です。  
``-h``オプションで使用できるオプションを確認してください。  
YOLOモデルファイルの指定(``-m_yolo``オプション)は必須です。  

### 簡易実行スクリプト
また、``demo.sh``を使用すると、入力ファイル指定だけでデフォルトのモデルファイルを使用して実行します。  
``demo.sh``の第1パラメータに``ncs``を指定すると、NCS2を使用して実行します。  
入力ファイルの後ろに指定したパラメータはyolo.pyにオプションとして渡されます。  

また、使用するモデルファイルは``MODLE_FILE``変数として指定できますので、使用したいモデルに合わせて設定してください。  
``MODLE_FILE``変数が設定されていない場合はスクリプト内で設定されている``default_model_file``変数が使用されます。  

```
==== usage ====
./demo.sh [ncs] input_file [other option(s)]
```

モデルファイルを指定する場合の例：  
```
MODEL_FILE=models/yolo_v4_tiny.xml ./demo.sh images/testimage.jpg 
```

### yolo.pyのオプション
```
positional arguments:
  INPUT_FILE            Path to the input video/picture

optional arguments:
  -h, --help            show this help message and exit

General:
  -o OUTPUT_FILE, --output OUTPUT_FILE
                        (optional) Path to save the output video to
  --crop WIDTH HEIGHT   (optional) Crop the input stream to this size
                        (default: no crop).
  -l LIB_PATH, --cpu_lib LIB_PATH
                        (optional) For MKLDNN (CPU)-targeted custom layers, if
                        any. Path to a shared library with custom layers
                        implementations
  -c LIB_PATH, --gpu_lib LIB_PATH
                        (optional) For clDNN (GPU)-targeted custom layers, if
                        any. Path to the XML file with descriptions of the
                        kernels
  -tl, --timelapse      (optional) Auto-pause after each frame
  --no_show             (optional) Do not display output
  -v, --verbose         (optional) Be more verbose
  -pc, --perf_stats     (optional) Output detailed per-layer performance stats
  --disable_status      (optional) Disable status frame

YOLO:
  -m_yolo MODEL_PATH    Path to the Face Detection model XML file
  -d_yolo {CPU,GPU,MYRIAD}
                        (optional) Target device for the Face Detection model
                        (default: CPU)
  -t_yolo [0..1]        (optional) Probability threshold for YOLO(default:
                        0.6)
  --iou_threshold IOU_THRESHOLD
                        (optional) Intersection over union threshold for
                        overlapping detections filtering(default: 0.4)
```
