#!/bin/bash

# USAGE
function usage () {
    echo -e "\n==== usage ===="
    echo -e "$0 [ncs] input_file [other option(s)]\n\n\n"
    exit 1
}

# スクリプト本体
cmd="face.py"

# 各モデルファイルの指定
opt="       -m_fd models/face-detection-retail-0004.xml"
opt="${opt} -m_lm models/landmarks-regression-retail-0009.xml"
opt="${opt} -m_hp models/head-pose-estimation-adas-0001.xml"

# 入力ファイル指定は必須
if [ $# -eq 0 ]; then
	# パラメータなしエラー
	usage
fi

# 最初のパラメータがncsだったらNCSを使用するオプションを追加
if [ ${1} = "ncs" ]; then
    opt="${opt} -d_fd MYRIAD"
    opt="${opt} -d_lm MYRIAD"
    opt="${opt} -d_hp MYRIAD"
	# "ncs"指定を捨てる
	shift
fi

# パラメータが"ncs"だけっだった場合の対策
if [ $# -eq 0 ]; then
	# パラメータなしエラー
	usage
fi

# 2番目以降すべてのパラメータを追加
opt="${opt} ${@:2}"

# 最初のパラメータは入力ファイル
opt="${opt} ${1}"
echo "python ${cmd} ${opt}"
python ${cmd} ${opt}
