for model in `ls models/*.xml`; do
	echo ${model}
	MODEL_FILE=${model} ./demo.sh ../sample/ssd_out.jpg
done
