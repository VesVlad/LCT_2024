import gc
import os
import requests
import torch
import openvino as ov
import typing as tp
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, Wav2Vec2Processor, Wav2Vec2Model, VivitImageProcessor, VivitModel
import logging
import av
import numpy as np
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()
core = ov.Core()
text_model_xml = "models/model.xml"
text_model = core.read_model(model=text_model_xml)
compiled_text_model = core.compile_model(model=text_model, device_name="CPU")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

av_model_xml = "models/model_av.xml"
av_model = core.read_model(model=av_model_xml)
compiled_av_model = core.compile_model(model=av_model, device_name="CPU")

wav_processor = Wav2Vec2Processor.from_pretrained("/tmp/model_audio")
wav_model = Wav2Vec2Model.from_pretrained("/tmp/model_audio")

image_processor = VivitImageProcessor.from_pretrained("/tmp/model_video")
vivit_model = VivitModel.from_pretrained("/tmp/model_video")


class UserRequestRelevants(BaseModel):
    query: str


class UserRequestLoad(BaseModel):
    description: str
    link: str


@app.post("/send_query")
def find_relevants(data: UserRequestRelevants):

    encoded_input = tokenizer(data.query, max_length=512,
                                   truncation=True, 
                                   padding='max_length', 
                                   return_tensors='pt')
    result = compiled_text_model({**encoded_input})

    query_emb = list(result.get(list(result.keys())[1]))

    query_emb = query_emb[0].tolist()
    data = {
        'message': str(query_emb)[1:-1]
    }
    res = requests.get(os.getenv('BASE_URL') + 'similar-videos', json=data)
    return res.json()
   

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
      if i in indices:
        frames.append(frame)
    return torch.concat([torch.from_numpy(x.to_ndarray(format="rgb24")).unsqueeze(0) for x in frames])


def parse(link: str) -> tp.Tuple[str, torch.Tensor, torch.Tensor]:
    fp = f'{link.split("/")[-2]}.mp4'
    r = requests.get(link, stream=True)
    with open(fp, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    container = av.open(fp)
    video = container.streams.video[0]
    seconds = int(video.duration * video.time_base)
    frames = seconds * video.average_rate.numerator
    indices = np.array([i * (frames // 32) for i in range(32)])

    video = read_video_pyav(container, indices)
    inputs_video = image_processor(list(video.permute(0, 3, 1, 2)), return_tensors="pt")
    inputs_video = {k: v for k, v in inputs_video.items()}
    container.seek(0)
    audio = []
    for frame in container.decode(audio=0):
        if len(frame.to_ndarray()) == 0:
            continue
        audio.append(torch.from_numpy(frame.to_ndarray()))
    audio = torch.concat(audio, axis=1)
    inputs_audio = wav_processor(audio[0], return_tensors="pt")

    with torch.no_grad():
      outputs_video = vivit_model(**inputs_video).pooler_output
      outputs_audio = wav_model(**inputs_audio).extract_features[:, 0, :]

    os.remove(fp)
    gc.collect()
    return link.split("/")[-2], outputs_video, outputs_audio


@app.post("/upload_video")
def upload_video(data: UserRequestLoad) -> int:
    name, outputs_video, outputs_audio = parse(data.link)
    print(outputs_video.shape, outputs_audio.shape)
    encoded_av_input = torch.concat([outputs_video, outputs_audio], axis=1)
    print(encoded_av_input.shape)
    processed = compiled_av_model(encoded_av_input)
    output_av = processed.get(list(processed.keys())[0]).squeeze(1)

    data = [
        {
            "id": uuid.uuid4().int,
            "link": data.link,
            "description": data.description,
            "vector": output_av.tolist()[0],
        }
    ]
    resp = requests.post(os.getenv('BASE_URL') + 'video', json=data)
    return resp.json()
