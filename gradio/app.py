import torch
from model import LSTMModel

import gradio as gr
import torch

model = torch.load("LSTM_AdamW.pt")

def classify_sentence(sentence):
    preds = model(sentence, pro = True, batched = False)
    dict = {"Negative": float(preds[0]),
            "Neutral": float(preds[1]),
            "Positive": float(preds[2])}
    return dict

demo = gr.Interface(fn=classify_sentence, 
            inputs=gr.Textbox(placeholder="Enter a sentence you'd like to classify here..."), 
            outputs="label",
            examples=[["This is wonderful!"], ["It's okayish."], ["The product is not bad."], ["Unacceptable quality!"]])

demo.launch()