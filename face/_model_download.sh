MODELS_DIR=models
MODELS_DIR_ABS=`realpath ${MODELS_DIR}`
mkdir -p ${MODELS_DIR}

# MODEL ZOOからのダウンロード ##################################################################
function download_model_zoo () {
# 	local BASE_URL="https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1"
	local BASE_URL="https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/"
	
	local MODEL_NAME=$1
	
	wget --no-check-certificate -O ${MODELS_DIR}/${MODEL_NAME}.bin ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
	wget --no-check-certificate -O ${MODELS_DIR}/${MODEL_NAME}.xml ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
	
}

download_model_zoo	"face-detection-retail-0004"
download_model_zoo 	"landmarks-regression-retail-0009"
download_model_zoo	"head-pose-estimation-adas-0001"



# MODELS_DIR=models
# mkdir -p ${MODELS_DIR}
# 
# BASE_URL="https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1"
# 
# 
# MODEL_NAME="face-detection-retail-0004"
# wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
# wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
# 
# MODEL_NAME="landmarks-regression-retail-0009"
# wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
# wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
# 
# MODEL_NAME="head-pose-estimation-adas-0001"
# wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
# wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
