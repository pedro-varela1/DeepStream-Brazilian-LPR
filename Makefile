all:
	@make -C nvinfer_custom_lpr_parser
	@make -C deepstream-lpr-app
	@make -C nvdsinfer_custom_impl_Yolo
clean:
	@make clean -C nvinfer_custom_lpr_parser
	@make clean -C deepstream-lpr-app
	@make clean -C nvdsinfer_custom_impl_Yolo
