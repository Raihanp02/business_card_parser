import os
os.environ["FLAGS_use_mkldnn"] = "0"

from app.pipelines.ocr_parser import OCRParser
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()
ocr_parser = OCRParser()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse-card")
def ocr_read(file: UploadFile = File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content))
    result = ocr_parser.process(image)
    return {"result": result}