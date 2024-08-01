FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /opt/KAIR

WORKDIR /opt/KAIR

CMD python main_train_dncnn.py

#CMD python main_test_dncnn.py --testset_name bsd68 --model_name epoch_8061_iter_395000_G --model_pool /opt/KAIR/denoising/dncnn25/models \
#--testsets /opt/KAIR/testsets --results /opt/KAIR/denoising/dncnn25/images

#CMD python generate_shifted_images_png.py

CMD sleep infinity
# Epoch 102: 27.50dB
# Epoch 2244: 28.67dB
#CMD sleep infinity
# TODO:
# https://arxiv.org/pdf/1902.02452

