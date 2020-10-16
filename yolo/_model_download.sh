MODELS_DIR=models
MODELS_DIR_ABS=`realpath ${MODELS_DIR}`
mkdir -p ${MODELS_DIR}

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
	local BASE_URL="https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/"
	
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
	${INTEL_OPENVINO_DIR}/deployment_tools/tools/model_downloader/downloader.py --name ${MODEL_NAME} --output_dir ${MODELS_DIR}
	
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
download_model_downloader  "yolo-v3-tiny-tf"       "coco"
download_model_downloader  "yolo-v2-tf"            "coco"
download_model_downloader  "yolo-v2-tiny-tf"       "coco"

download_and_convert_v3_tiny		# for yolo_v3_tiny
download_and_convert_v4				# for yolo_v4
download_and_convert_v4_tiny		# for yolo_v4_tiny

<< COMMENT
【memo】============================================================
download_and_convert_XXXを実行する際は最低限以下のモジュールが必要
(openVINOのインストールが完了してれば問題なし)
------------------------------------
absl-py==0.10.0
astor==0.8.1
decorator==4.4.2
defusedxml==0.6.0
gast==0.2.2
google-pasta==0.2.0
grpcio==1.32.0
h5py==2.10.0
importlib-metadata==2.0.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
Markdown==3.3
networkx==2.5
numpy==1.18.5
opt-einsum==3.3.0
Pillow==7.2.0
protobuf==3.13.0
six==1.15.0
tensorboard==1.15.0
tensorflow==1.15.4
tensorflow-estimator==1.15.1
termcolor==1.1.0
Werkzeug==1.0.1
wrapt==1.12.1
zipp==3.3.0
------------------------------------

以下を実行すればインストールされるはず
pip install tensorflow==1.15.4
pip install Pillow
pip install networkx defusedxml

COMMENT
