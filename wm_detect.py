import numpy as np
from astropy.timeseries import LombScargle
import hashlib
from typing import List

import torch
from transformers import LogitsWarper
import torch.nn.functional as F
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import matplotlib.pyplot as plt  
from transformers import T5ForConditionalGeneration, T5Tokenizer

from llama.tokenizer import Tokenizer

class WatermarkBase:
    """
    Base class for watermarking distributions.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
        freq: The frequency of the watermark.
        eps: The epsilon value for the watermark.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 1234,
                 freq: int = 16, eps: float = 0.2):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.vec = torch.tensor(rng.normal(loc=0, scale=1, size=(vocab_size, 256)))
        self.key = torch.tensor(rng.random(256))
        self.freq = freq
        self.eps = eps
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class WatermarkLogitsWarper(WatermarkBase, LogitsWarper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        x = torch.matmul(self.vec[input_ids], self.key)
        x_ = torch.distributions.Normal(0, 1).cdf(x / np.sqrt(self.key.shape[0] / 3))
        probs = F.softmax(scores[0], dim=0)
        g0_idx = torch.where(self.green_list_mask == 0)[0]
        g1_idx = torch.where(self.green_list_mask == 1)[0]
        g0_prob = probs[g0_idx].sum()

        g1_prob = probs[g1_idx].sum()
        z0 = torch.cos(self.freq * x_)
        z1 = torch.cos(torch.add(self.freq * x_, torch.pi))
        new_g0_prob = (g0_prob + 1e-25 + self.eps * (1 + z0)) / (1 + 2 * self.eps + 1e-25)
        new_g1_prob = (g1_prob + 1e-25 + self.eps * (1 + z1)) / (1 + 2 * self.eps + 1e-25)
        new_probs = probs.clone()
        new_probs[g0_idx] = new_g0_prob / g0_prob * probs[g0_idx]
        new_probs[g1_idx] = new_g1_prob / g1_prob * probs[g1_idx]
        new_logits = torch.log(new_probs).view(1, -1)
        return new_logits

class WatermarkDetector:
    def __init__(self, suspect_model, vocab_size, green_list_fraction, watermark_key, freq, eps):
        self.hash_fn = WatermarkBase._hash_fn
        rng = np.random.default_rng(self.hash_fn(watermark_key))
        mask = np.array([True] * int(green_list_fraction * vocab_size) + [False] * (vocab_size - int(green_list_fraction * vocab_size)))
        rng.shuffle(mask)
        self.vec = torch.tensor(rng.normal(loc=0, scale=1, size=(vocab_size, 256))).to('cuda')
        self.key = torch.tensor(rng.random(256)).to('cuda')
        self.suspect_model = suspect_model
        self.vocab_size = vocab_size
        self.green_list_fraction = green_list_fraction
        self.watermark_key = watermark_key
        self.freq = freq
        self.eps = eps
        self.H = set()

        
        self.watermark_base = WatermarkBase(
            fraction=self.green_list_fraction,
            strength=None,  # Strength isn't needed for detection
            vocab_size=self.vocab_size,
            watermark_key=self.watermark_key,
            freq=self.freq,
            eps=self.eps
        )

    def detect(self, probing_data):
        attack_tokenizer = T5Tokenizer.from_pretrained('t5_model')
        llama_tokenizer = Tokenizer("/data/haoxing/codellama/CodeLlama-7b/tokenizer.model")
        for i, x in enumerate(probing_data): # x here is the prompt text
            print(f'Adding prompt {i} to H begins.')
            #import pdb; pdb.set_trace()
            '''
            input_ids = attack_tokenizer.encode(x, return_tensors='pt').to(self.suspect_model.device)
            with torch.no_grad():
                outputs = self.suspect_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
                print(attack_tokenizer.decode(outputs[0], skip_special_tokens=True))
            '''
            inputs = attack_tokenizer(x, max_length=256, padding="max_length", truncation=True, add_special_tokens=True)
    
            input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0)

            outputs = self.suspect_model.generate(input_ids=input_ids.to('cuda:0'), attention_mask=attention_mask.to('cuda:0'))
            print(attack_tokenizer.decode(outputs.flatten(), skip_special_tokens=True))
            for j, token_id in enumerate(outputs[0]):
                if j == 0:
                    x_ids = llama_tokenizer.encode(x, bos=True, eos=False)[-1] # the last token from the input x
                    print(x_ids)
                else:
                    # should push outputs[0][j-1] instead of token_id
            
                    completion = attack_tokenizer.decode(outputs.flatten()[j-1], skip_special_tokens=True)
                    if completion != '':
                    #import pdb; pdb.set_trace()
                        x_ids = llama_tokenizer.encode(completion, bos=False, eos=False)[0]
                        print(x_ids)
                

                t_ = torch.matmul(self.vec[x_ids], self.key)
                t = torch.distributions.Normal(0, 1).cdf(t_ / np.sqrt(self.key.shape[0] / 3))

                g0_idx = torch.where(self.watermark_base.green_list_mask == 0)[0].to('cuda')
                
                if x_ids in g0_idx:
                    self.H.add((t.cpu(), 1))
                else:
                    self.H.add((t.cpu(), 0))

            print(f'Adding prompt {i} to H done.')
        times, signal = [], []
        for time, sig in self.H:
            times.append(time)
            signal.append(sig)



        all_times = np.array(times)
        all_signal = np.array(signal)

        frequency, power = LombScargle(all_times, all_signal).autopower()
        plt.plot(frequency, power)
        plt.xscale('log')
        plt.axvline(x=16, color='r', linestyle='--')
        plt.show()
        plt.savefig("Lomb.png")
        print(f"frequency: {frequency}")
        P_snr = self.compute_snr(frequency, power)

        return P_snr

    def compute_snr(self, frequency, power):
        # Define the frequency window around the watermark frequency
        delta = 5  # This is an example value for δ, adjust as appropriate
        fw = self.freq  # The watermark frequency

        # Calculate P_signal
        signal_frequency_mask = (frequency >= (fw - delta / 2)) & (frequency <= (fw + delta / 2))
        P_signal = np.trapz(power[signal_frequency_mask], frequency[signal_frequency_mask]) / delta

        # Calculate P_noise
        noise_frequency_mask = ~signal_frequency_mask
        P_noise = np.trapz(power[noise_frequency_mask], frequency[noise_frequency_mask]) / (frequency.max() - delta)

        # Calculate SNR
        print(f"P_signal: {P_signal}")
        print(f'P_noise: {P_noise}')
        P_snr = P_signal / P_noise

        return P_snr
