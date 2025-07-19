from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import json

class Singleton:
    def __init__(self, cls):
        self._cls = cls
    def instance(self, **kwargs):
        try:
            return self._instance
        except:
            self._instance = self._cls(**kwargs)
            return self._instance

def load_config(config_path):
    def wrapper(cls):
        cls.config = json.load(config_path)
        return cls
    return wrapper

def load_pipeline(modelpath):
    pipe = pipeline("text-generation", model=modelpath, torch_dtype=torch.bfloat16, device_map="auto")
    return pipe

def load_model(modelpath, local_files_only=False, trust_remote_code=False):
    model = AutoModel.from_pretrained(modelpath, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    return model

def load_tokenizer(modelpath):
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    return tokenizer


# "Tscontriever": {"model_name_or_path":"./models/Tscontriever"},

class ContrieverEmbedder:
    def __init__(self, model_name_or_path, device="cpu"):
        self.device = device
        print(f"Initializing ContrieverEmbedder with model: {model_name_or_path} on device: {self.device}")
        # Pass trust_remote_code to load_model
        self.model = load_model(model_name_or_path)
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode (good practice for inference)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def batch_embedding(self, contents, return_type = 'nparray'):
        inputs = self.tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = self.model(**inputs)

        outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        if return_type == 'nparray':
            return embeddings.cpu().detach().numpy()
        else:
            return embeddings.cpu().detach().numpy().tolist()

    def embedding(self, content, return_type = 'nparray'):
        return self.batch_embedding([content], return_type=return_type)[0]