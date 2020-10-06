MODELS_DIR=models
mkdir -p ${MODELS_DIR}

BASE_URL="https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1"


MODEL_NAME="face-detection-retail-0004"
wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml

MODEL_NAME="landmarks-regression-retail-0009"
wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml

MODEL_NAME="head-pose-estimation-adas-0001"
wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.bin
wget --no-check-certificate -P ${MODELS_DIR} ${BASE_URL}/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
