from app.ai_services.ner import CustomNER
from app.ai_services.ocr.ocr import OCR

class OCRParser:
    def __init__(self, ocr = OCR(), nlp = CustomNER()):
        self.ocr = ocr
        self.nlp = nlp

    def process(self, image):
        sentences = self.ocr.process(image)
        result = self.nlp.process(sentences)

        return result