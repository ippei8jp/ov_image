MODELS_DIR=models
MODELS_DIR_ABS=`realpath ${MODELS_DIR}`
mkdir -p ${MODELS_DIR}

# labelsファイルのダウンロード
function download_labels_master () {
	# YOLO向けファイルとは違うので注意!！
	local LABEL_URL="https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt"
	local LABEL_NAME="coco"
	wget -O ${MODELS_DIR}/${LABEL_NAME}.labels  ${LABEL_URL}
	# ID=0 の分を追加
	sed -i '1s/^/BLANK\n/' ${MODELS_DIR}/${LABEL_NAME}.labels
}

# model_downloaderを使用したダウンロード ##################################################################
function download_model_downloader () {
	local MODEL_NAME=$1
	local LABEL_NAME=$2
	
	# 元モデルのダウンロード
	### # --output_dirオプションの動きが変わった？？
	### # とりあえずカレントにダウンロードするようにして回避
	### ${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py --name ${MODEL_NAME} --output_dir ${MODELS_DIR}
	pushd ${MODELS_DIR}
	${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py --name ${MODEL_NAME}
	popd
	
	# IRモデルへの変換
	${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/converter.py --precisions FP16 --name ${MODEL_NAME} --download_dir  ${MODELS_DIR} --output_dir ${MODELS_DIR}
	# mv ${MODELS_DIR}/public/${MODEL_NAME}/FP16/${MODEL_NAME}.{xml,bin} ${MODELS_DIR}
	ln -sf `realpath --relative-to=${MODELS_DIR} ${MODELS_DIR}/public/${MODEL_NAME}/FP16/${MODEL_NAME}.{xml,bin}` ${MODELS_DIR}
	
	# LABELファイルへのリンク作成
	ln -sf `realpath --relative-to=${MODELS_DIR} ${MODELS_DIR}/${LABEL_NAME}.labels` ${MODELS_DIR}/${MODEL_NAME}.labels
}


download_labels_master

download_model_downloader  "ssd_mobilenet_v2_coco"            "coco"
