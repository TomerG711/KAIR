FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /opt/KAIR

WORKDIR /opt/KAIR

CMD python main_train_dncnn.py

#CMD sleep infinity
# Epoch 102: 27.50dB
# Epoch 2244: 28.67dB