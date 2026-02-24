import torch
from pathlib import Path
import gdown
from tokenizers import Tokenizer
from tokenizers.processors import RobertaProcessing
from collections import defaultdict


class CustomNER:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path(__file__).parent.resolve().parents[1]

    def __init__(self, 
                 model_path="assets/models/xlmroberta.pt", 
                 tokenizer_path="assets/models/xlmr_tokenizer/tokenizer.json"):
        self.id2label= {0: 'O', 1: 'B-Phone', 2: 'I-Phone', 3: 'B-Email', 4: 'I-Email', 5: 'B-Person', 6: 'I-Person', 7: 'B-Company', 8: 'I-Company', 9: 'B-Position', 10: 'I-Position', 11: 'B-Link', 12: 'I-Link', 13: 'B-Address', 14: 'I-Address'}
        self.model_dir = self.BASE_DIR / Path(model_path)
        self.tokenizer_dir = self.BASE_DIR / Path(tokenizer_path)

        self._load_model()
        self._load_tokenizer()

    def process(self, sentences):
        result = {"Company": [], "Phone": [], "Email": [], "Link": [], "Person": [], "Position": [], "Address": []}

        for text in sentences:
            tokens, pred_ids = self.predict_tokens(text)
            self.postprocess(tokens, pred_ids, result)

        return result
    
    def predict_tokens(self, sentence):
        # Tokenize
        encoded = self.tokenizer.encode(sentence)
        input_ids = torch.tensor([encoded.ids])
        attention_mask = torch.tensor([encoded.attention_mask])

        # Inference
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        logits = output["logits"]
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        tokens = encoded.tokens

        return tokens, pred_ids
    
    def postprocess(self, tokens, pred_ids, result: dict):
        aggregated_entities = defaultdict(list)
        current_tokens = []
        current_entity_type = None

        for token, pred in zip(tokens, pred_ids):
            if token in ("<s>", "</s>"):
                continue 
            label = self.id2label[pred]

            if label == "O":
                self._flush_entity(aggregated_entities, current_tokens, current_entity_type)
                current_tokens = []
                current_entity_type = None
                continue

            if "-" in label:
                _, ent_type = label.split("-")
            else:
                ent_type = label

            if current_entity_type == ent_type:
                current_tokens.append(token)
            else:
                self._flush_entity(aggregated_entities, current_tokens, current_entity_type)
                current_entity_type = ent_type
                current_tokens = [token]

        # Final flush
        self._flush_entity(aggregated_entities, current_tokens, current_entity_type)

        # Print selected entity types
        for key in ["Company", "Phone", "Email", "Link", "Person", "Position", "Address"]:
            if key in aggregated_entities:
                result[key].append(aggregated_entities[key][0])
    
    def _flush_entity(self, aggregated_entities, current_tokens, current_entity_type):
        if current_tokens and current_entity_type:
            word = "".join(current_tokens).replace("▁", " ").strip()
            if word:
                aggregated_entities[current_entity_type].append(word)

    def _is_sublist_in_order(self,sub, main):
        n, m = len(sub), len(main)
        for i in range(m - n + 1):  # Sliding window approach
            if main[i:i+n] == sub:
                return i, n
        return False

    def _reassin(self,a,b):
        b1 = []
        for i in range(len(b)):
            b1.append(self.tokenizer.tokenize(b[i]))

        labeltemp = []
        for i in b1:
            labeltemp.append([0]*len(i))

        for i in a:
            categ1 = self.model.config.label2id[f'B-{i}']
            categ2 = self.model.config.label2id[f'I-{i}']

            for j in a[i]:
                temp = self.tokenizer.tokenize(j)

            for k in range(len(b1)):
                result = self._is_sublist_in_order(temp,b1[k])

                if result:
                    l,n = result
                    if labeltemp[k][l] == 0:
                        labeltemp[k][l] = categ1
                        labeltemp[k][l+1:l+n] = [categ2] * (n-1)
                        break
                    else:
                        break

                if k == len(b)-1:
                    return j

        return labeltemp

    def _rearrange(self,a):
        for i in a:
            a[i].sort(key= lambda x: len(x), reverse=True)
        return a
        
    def _load_model(self):
        if not self.model_dir.parents[0].exists():
            self.model_dir.parents[0].mkdir(parents=True, exist_ok=True)

        if not self.model_dir.exists():
            print("model not found, downloading...")
            gdown.download(id='160vL5_jhXjJK04J-EQ5ahw10zCJ_ni8t', output=str(self.model_dir.parent), quiet=False, use_cookies=False)

        self.model = torch.jit.load(str(self.model_dir)).eval().to(self.device)

    def _load_tokenizer(self):
        if not self.tokenizer_dir.exists():
            print("tokenizer not found, downloading...")
            gdown.download_folder(id='13Jj_RlQkbcCO_R1LkjxejlvonEMQg4DR', output=str(self.tokenizer_dir.parent), quiet=False, use_cookies=False)

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_dir))

        self.tokenizer.post_processor = RobertaProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        self.tokenizer.enable_truncation(max_length=256)