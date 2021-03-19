MODELS_DIR=models
MODELS_DIR_ABS=`realpath ${MODELS_DIR}`
mkdir -p ${MODELS_DIR}

# openVINOのバージョン情報
OV_VER_STR=$(echo `realpath $INTEL_OPENVINO_DIR` | sed -e "s/^.*openvino_\(.*\..*\)\..*$/\1/g")         # 2020.3 など
OV_VER_MAJOR=$(echo $OV_VER_STR | sed -e "s/\(.*\)\..*/\1/g")                                              # 2020 など
OV_VER_MINOR=$(echo $OV_VER_STR | sed -e "s/.*\.\(.*\)/\1/g")                                              # 3 など

# MODEL ZOOからのダウンロード ##################################################################
function download_model_zoo () {
if [ ${OV_VER_MAJOR} -eq 2020 ] ; then
	local BASE_URL="https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1"
else
#	local BASE_URL="https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/"
	local BASE_URL="https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/"
fi

	local MODEL_NAME=$1

	wget --no-check-certificate -O ${MODELS_DIR}/${MODEL_NAME}.bin ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
	wget --no-check-certificate -O ${MODELS_DIR}/${MODEL_NAME}.xml ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
	
}

download_model_zoo	"face-detection-retail-0004"
download_model_zoo 	"landmarks-regression-retail-0009"
download_model_zoo	"head-pose-estimation-adas-0001"
