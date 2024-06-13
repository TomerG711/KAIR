FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /opt/KAIR

WORKDIR /opt/KAIR

CMD python main_train_druent.py

#CMD sleep infinity