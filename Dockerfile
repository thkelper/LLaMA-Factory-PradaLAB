FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt \
    -i https://mirrors.baidu.com/pypi/simple/

COPY . /app/
RUN pip install -e .[deepspeed,metrics,bitsandbytes,qwen] \
    -i https://mirrors.baidu.com/pypi/simple/

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860

CMD [ "llamafactory-cli", "webui" ]
