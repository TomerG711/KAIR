FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /opt/KAIR

WORKDIR /opt/KAIR

#CMD python main_train_dncnn.py

CMD python main_test_dncnn.py --testset_name bsd68 --model_name 100000_G --model_pool /opt/KAIR/denoising/dncnn25/models \
--testsets /opt/KAIR/testsets --results /opt/KAIR/denoising/dncnn25/images

#CMD sleep infinity
# Epoch 102: 27.50dB
# Epoch 2244: 28.67dB