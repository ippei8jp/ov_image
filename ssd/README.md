# SSD処理

## ファイル構成

| ファイル               | 内容                                             |
|------------------------|--------------------------------------------------|
| demo.sh                | スクリプト実行コマンド                           |
| _model_download.sh     | モデルファイルをダウンロードするスクリプト       |
|                        |                                                  |
| ssd.py                 | 顔検出＆関連検出スクリプト本体                   |
| ssd_frame_processor.py | フレーム毎の処理クラス                           |
| ssd_visualizer.py      | 画像関連管理クラス                               |
|                        |                                                  |
| ssd_detector.py        | SSD検出器クラス                                 |


## SSD処理プログラム実行方法

あらかじめ`` bash _model_download.sh``を使用してモデルファイルをダウンロードしてください。  
``public``ディレクトリに元となるTensorflow用モデルをダウンロードし、
openVINO用モデルに変換したものを``models``ディレクトリに格納します。  
openVINO用モデルに変換した後は``public``ディレクトリは不要ですので削除してもかまいません。  

ssd.pyが処理本体です。  
``-h``オプションで使用できるオプションを確認してください。  
SSDモデルファイルの指定(``-m_ssd``オプション)は必須です。  


### 簡易実行スクリプト
また、``demo.sh``を使用すると、入力ファイル指定だけでデフォルトのモデルファイルを使用して実行します。  
``demo.sh``の第1パラメータに``ncs``を指定すると、NCS2を使用して実行します。  
入力ファイルの後ろに指定したパラメータはssd.pyにオプションとして渡されます。  

```
==== usage ====
./demo.sh [ncs] input_file [other option(s)]
```

### ssd.pyのオプション
```
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

SSD:
  -m_ssd MODEL_PATH     Path to the Face Detection model XML file
  -d_ssd {CPU,GPU,MYRIAD}
                        (optional) Target device for the Face Detection model
                        (default: CPU)
  -t_ssd [0..1]         (optional) Probability threshold for SSD(default: 0.6)
```
