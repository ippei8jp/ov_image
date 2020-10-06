MODELS_DIR=models
mkdir -p ${MODELS_DIR}

# 使用するモデル名
modelname=ssd_mobilenet_v2_coco

# 元モデルのダウンロード
${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py --name ${modelname}

# IRモデルへの変換
${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/converter.py --precisions FP16 --name ${modelname}

# IRモデルファイルへのシンボリックリンク作成
# ln -sf ../public/${modelname}/FP16/${modelname}.{xml,bin} .
ln -sf `realpath --relative-to=${MODELS_DIR} public/${modelname}/FP16/${modelname}.{xml,bin}` ${MODELS_DIR}

# ラベルファイルのダウンロード
wget -O - https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_complete_label_map.pbtxt \
	| grep display_name | sed -e "s/^.*\"\(.*\)\".*$/\1/g" >  ${MODELS_DIR}/${modelname}.labels
