FROM python:3.11-slim-buster

WORKDIR ./server

RUN apt-get update && apt-get -y install git-lfs build-essential gcc python3-dev
RUN git lfs install

RUN git clone https://huggingface.co/facebook/wav2vec2-base-960h /tmp/model_audio
RUN git clone https://huggingface.co/google/vivit-b-16x2-kinetics400 /tmp/model_video

RUN git clone https://git.ffmpeg.org/ffmpeg.git

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . ./

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "80"]