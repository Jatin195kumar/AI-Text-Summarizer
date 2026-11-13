
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

model.to(device)
model.eval()

# Project directory
BASE_DIR = Path(__file__).resolve().parent


class DialogueInput(BaseModel):
    dialogue: str


# Clean input text
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    # Tokenization
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate summary
    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Decode output
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# API endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(dialogue_input.dialogue)

    return {"summary": summary}


# Homepage
@app.get("/")
async def home():

    html_file = BASE_DIR / "index.html"

    return FileResponse(html_file)
