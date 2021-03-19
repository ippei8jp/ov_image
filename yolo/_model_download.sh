MODELS_DIR=models
MODELS_DIR_ABS=`realpath ${MODELS_DIR}`
mkdir -p ${MODELS_DIR}

# openVINOのバージョン情報
OV_VER_STR=$(echo `realpath $INTEL_OPENVINO_DIR` | sed -e "s/^.*openvino_\(.*\..*\)\..*$/\1/g")         # 2020.3 など
OV_VER_MAJOR=$(echo $OV_VER_STR | sed -e "s/\(.*\)\..*/\1/g")                                              # 2020 など
OV_VER_MINOR=$(echo $OV_VER_STR | sed -e "s/.*\.\(.*\)/\1/g")                                              # 3 など

# labelsファイルのダウンロード
function download_labels_master () {
	local LABEL_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
	# 以下でも大丈夫(さがせばあちこちにあるみたい)
	# local LABEL_URL="https://raw.githubusercontent.com/david8862/keras-YOLOv3-model-set/master/configs/coco_classes.txt"
	local LABEL_NAME="coco"
	wget -O ${MODELS_DIR}/${LABEL_NAME}.labels  ${LABEL_URL}
	
	local LABEL_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names"
	LABEL_NAME="voc"
	wget -O ${MODELS_DIR}/${LABEL_NAME}.labels  ${LABEL_URL}
}

# MODEL ZOOからのダウンロード ##################################################################
function download_model_zoo () {
# 	local BASE_URL="https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1"
# 	local BASE_URL="https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/"
	local BASE_URL="https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/"
	
	local MODEL_NAME=$1
	local LABEL_NAME=$2
	
	wget --no-check-certificate -O ${MODELS_DIR}/${MODEL_NAME}.bin ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
	wget --no-check-certificate -O ${MODELS_DIR}/${MODEL_NAME}.xml ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
	
	# LABELファイルへのリンク作成
	ln -sf `realpath --relative-to=${MODELS_DIR} ${MODELS_DIR}/${LABEL_NAME}.labels` ${MODELS_DIR}/${MODEL_NAME}.labels
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

# 手動でコンバート処理(YOLO_V3_TINY) ##################################################################
# tf 2.xでは実行できない
function download_and_convert_v3_tiny () {
	local MODEL_NAME="yolo_v3_tiny"
	local CONVERTER_REPOSITORY="https://github.com/mystic123/tensorflow-yolo-v3.git"
	local CONVERTER_DIR="tensorflow-yolo-v3"
	local CHECKOUT_UID="ed60b90"
	local MODEL_DOWNLOAD_URL="https://pjreddie.com/media/files/yolov3-tiny.weights"
	local TRANS_CONFIG="${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/extensions/front/tf/${MODEL_NAME}.json"
	local LABEL_NAME="coco"

	# 作業ディレクトリ ${MODELS_DIR} へ移動
	pushd ${MODELS_DIR}
	
	if [ ! -d ${CONVERTER_DIR} ]; then
		# Darknet→tensorflowモデル変換プログラムの取得
		git clone ${CONVERTER_REPOSITORY}
		
		# 対象バージョンをcheck out
		cd ${CONVERTER_DIR}/
		git checkout ${CHECKOUT_UID}
	else
		cd ${CONVERTER_DIR}/
	fi
	
	# Darknet用モデルファイルのダウンロード
	wget -O ${MODEL_NAME}.weights ${MODEL_DOWNLOAD_URL}
	
	# Darknet→tensorflowモデル変換
	python convert_weights_pb.py \
		--class_names ${MODELS_DIR_ABS}/${LABEL_NAME}.labels \
		--data_format NHWC \
		--weights_file ${MODEL_NAME}.weights \
		--tiny
	mv frozen_darknet_yolov3_model.pb ${MODEL_NAME}.pb
	
	# tensorflow→openVINOモデル変換
	python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_tf.py \
	--input_model ${MODEL_NAME}.pb \
	--tensorflow_use_custom_operations_config ${TRANS_CONFIG} \
	--output_dir ${MODELS_DIR_ABS} \
	--batch 1 \
	--data_type FP16
	
	# 作業ディレクトリから復帰
	popd
	
	# LABELファイルへのリンク作成
	ln -sf `realpath --relative-to=${MODELS_DIR} ${MODELS_DIR}/${LABEL_NAME}.labels` ${MODELS_DIR}/${MODEL_NAME}.labels
}

# 手動でコンバート処理(YOLO_V4) ##################################################################
# openVINO 2021.1以降が必要
# tf 2.xでは実行できない
function download_and_convert_v4 () {
	local MODEL_NAME="yolo_v4"
	local CONVERTER_REPOSITORY="https://github.com/TNTWEN/OpenVINO-YOLOV4.git"
	local CONVERTER_DIR="OpenVINO-YOLOV4"
	local CHECKOUT_UID="7947ce4ca4e6602947484d98585abb013a422ece"
	local MODEL_DOWNLOAD_URL="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
	local TRANS_CONFIG="yolov4.json"
	local LABEL_NAME="coco"

	# 作業ディレクトリ ${MODELS_DIR} へ移動
	pushd ${MODELS_DIR}

	if [ ! -d ${CONVERTER_DIR} ]; then
		# Darknet→tensorflowモデル変換プログラムの取得
		git clone ${CONVERTER_REPOSITORY}
		
		# 対象バージョンをcheck out
		cd ${CONVERTER_DIR}/
		git checkout ${CHECKOUT_UID}
	else
		cd ${CONVERTER_DIR}/
	fi
	
	# Darknet用モデルファイルのダウンロード
	wget -O ${MODEL_NAME}.weights ${MODEL_DOWNLOAD_URL}
	
	# Darknet→tensorflowモデル変換
	python convert_weights_pb.py \
		--class_names ${MODELS_DIR_ABS}/${LABEL_NAME}.labels \
		--data_format NHWC \
		--weights_file ${MODEL_NAME}.weights
	mv frozen_darknet_yolov4_model.pb ${MODEL_NAME}.pb
	
	# tensorflow→openVINOモデル変換
	${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
	  --input_model ${MODEL_NAME}.pb \
	  --transformations_config ${TRANS_CONFIG} \
	  --reverse_input_channels \
	  --output_dir ${MODELS_DIR_ABS} \
	  --batch 1 \
	  --data_type FP16
	
	# 作業ディレクトリから復帰
	popd
	
	# LABELファイルへのリンク作成
	ln -sf `realpath --relative-to=${MODELS_DIR} ${MODELS_DIR}/${LABEL_NAME}.labels` ${MODELS_DIR}/${MODEL_NAME}.labels
}

# 手動でコンバート処理(YOLO_V4_TINY) ##################################################################
# openVINO 2021.1以降が必要
# tf 2.xでは実行できない
function download_and_convert_v4_tiny () {
	local MODEL_NAME="yolo_v4_tiny"
	local CONVERTER_REPOSITORY="https://github.com/TNTWEN/OpenVINO-YOLOV4.git"
	local CONVERTER_DIR="OpenVINO-YOLOV4"
	local CHECKOUT_UID="7947ce4ca4e6602947484d98585abb013a422ece"
	local MODEL_DOWNLOAD_URL="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
	local TRANS_CONFIG="yolo_v4_tiny.json"
	local LABEL_NAME="coco"

	# 作業ディレクトリ ${MODELS_DIR} へ移動
	pushd ${MODELS_DIR}

	if [ ! -d ${CONVERTER_DIR} ]; then
		# Darknet→tensorflowモデル変換プログラムの取得
		git clone ${CONVERTER_REPOSITORY}
		
		# 対象バージョンをcheck out
		cd ${CONVERTER_DIR}/
		git checkout ${CHECKOUT_UID}
	else
		cd ${CONVERTER_DIR}/
	fi
	
	# Darknet用モデルファイルのダウンロード
	wget -O ${MODEL_NAME}.weights ${MODEL_DOWNLOAD_URL}
	
	# Darknet→tensorflowモデル変換
	python convert_weights_pb.py \
		--class_names ${MODELS_DIR_ABS}/${LABEL_NAME}.labels \
		--data_format NHWC \
		--weights_file ${MODEL_NAME}.weights \
		--tiny
	mv frozen_darknet_yolov4_model.pb ${MODEL_NAME}.pb
	
	# tensorflow→openVINOモデル変換
	${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
	  --input_model ${MODEL_NAME}.pb \
	  --transformations_config ${TRANS_CONFIG} \
	  --reverse_input_channels \
	  --output_dir ${MODELS_DIR_ABS} \
	  --batch 1 \
	  --data_type FP16
	
	# 作業ディレクトリから復帰
	popd
	
	# LABELファイルへのリンク作成
	ln -sf `realpath --relative-to=${MODELS_DIR} ${MODELS_DIR}/${LABEL_NAME}.labels` ${MODELS_DIR}/${MODEL_NAME}.labels
}

download_labels_master

download_model_zoo         "yolo-v2-tiny-ava-0001" "voc"
download_model_zoo         "yolo-v2-ava-0001"      "voc"

download_model_downloader  "yolo-v3-tf"            "coco"
if [ ${OV_VER_MAJOR} -ge 2021 ] ; then
	# 以下は2020では存在しない
	download_model_downloader  "yolo-v3-tiny-tf"       "coco"
fi
download_model_downloader  "yolo-v2-tf"            "coco"
download_model_downloader  "yolo-v2-tiny-tf"       "coco"

download_and_convert_v3_tiny		# for yolo_v3_tiny
download_and_convert_v4_tiny		# for yolo_v4_tiny
if [ ${OV_VER_MAJOR} -ge 2021 ] ; then
	# 以下は2020ではエラーになる
	download_and_convert_v4				# for yolo_v4
fi

<< COMMENT
【memo】============================================================
download_and_convert_XXXを実行する際は
openVINOのインストールが完了し他状態でpillowをインストールする必要がある

pip install Pillow

COMMENT
