#!/bin/bash

# USAGE
function usage () {
    echo -e "\n==== usage ===="
    echo -e "$0 [ncs] input_file [other option(s)]\n\n\n"
    exit 1
}

# スクリプト本体
cmd="yolo.py"

# デフォルトのモデルファイル
# default_model_file="models/yolo-v2-ava-0001.xml"
# default_model_file="models/yolo-v2-tf.xml"
# default_model_file="models/yolo-v2-tiny-ava-0001.xml"
# default_model_file="models/yolo-v2-tiny-tf.xml"
# default_model_file="models/yolo-v3-tf.xml"
default_model_file="models/yolo-v3-tiny-tf.xml"
# default_model_file="models/yolo_v3_tiny.xml"
# default_model_file="models/yolo_v4.xml"
# default_model_file="models/yolo_v4_tiny.xml"


# モデルファイルの指定
opt="       -m_yolo ${MODEL_FILE:=$default_model_file}"

# 入力ファイル指定は必須
if [ $# -eq 0 ]; then
	# パラメータなしエラー
	usage
fi

# 最初のパラメータがncsだったらNCSを使用するオプションを追加
if [ ${1} = "ncs" ]; then
    opt="${opt} -d_ssd MYRIAD"
	# "ncs"指定を捨てる
	shift
fi

# パラメータが"ncs"だけっだった場合の対策
if [ $# -eq 0 ]; then
	# パラメータなしエラー
	usage
fi

# openvino-python環境で実行するときのために、LD_LIBRARY_PATHを追加
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYENV_ROOT}/versions/`pyenv version-name`/lib

# 2番目以降すべてのパラメータを追加
opt="${opt} ${@:2}"

# 最初のパラメータは入力ファイル
opt="${opt} ${1}"
echo "python ${cmd} ${opt}"
python ${cmd} ${opt}
