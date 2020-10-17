for model in `ls models/*.xml`; do
	echo ${model}
	MODEL_FILE=${model} ./demo.sh images/teddybear.jpg 
done
