import torch
from wm_detect import WatermarkDetector
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_path = 't5_model'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

watermark_detector = WatermarkDetector(
    suspect_model= model,
    green_list_fraction=0.5,  # Example fraction of green-listed tokens
    #strength=2.0,  # Example strength of the watermark
    vocab_size=50257,  # Example vocabulary size
    watermark_key=1234,  # The seed used in the watermarking process
    freq=16,  # The frequency of the watermark
    eps=0.2  # The epsilon value for the watermark
)

probing_data = torch.load('probing_data.pt')
psnr = watermark_detector.detect(probing_data)
print(psnr)