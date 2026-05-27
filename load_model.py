import torch
from sae_lens import HookedSAETransformer, SAE
from dotenv import load_dotenv
import os
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import json
import pandas as pd

repo_id = "beniaminbrad/yellow_goblin_gemma"

def huggingface_login():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found! Check .env file.")
    login(token=HF_TOKEN)
    print("Logged in to HuggingFace successfully!")
def load_model(model_name, sae_release, sae_id, device = "cuda"):
    '''
    Loads the LLM and the SAE. Returns both.'''
    huggingface_login() # Log in to HF
    print("Loading Model...")
    sae_model = HookedSAETransformer.from_pretrained_no_processing(
        model_name, # Assuming you switch to Gemma for the official SAEs
        device=device,
        dtype=torch.bfloat16 # Standard precision for interpretability!
    )

    # 2. Load the Pre-trained SAE
    # This example loads a Gemma Scope SAE for Layer 12
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device
    )
    
    setattr(sae, 'fold_W_dec_norm', lambda: None) 
    print(f"Loaded model: {model_name}")
    print(f"Loaded SAE: {sae_id} from release {sae_release}")
    return sae_model, sae
def load_feature_titles(repo_id = repo_id):
    print("Downloading feature titles from Hugging Face...")
    
    # This downloads the file and caches it locally so it's instant next time
    file_path = hf_hub_download(repo_id=repo_id, filename="feature_titles.json")

    with open(file_path, "r") as f:
        raw_dict = json.load(f)
    feature_titles = {int(k): v for k, v in raw_dict.items()}
    
    print("Successfully loaded feature titles.")
    return feature_titles

def load_sample_dataset(repo_id = repo_id, num_samples = 5000):
    print("Downloading dataset from Hugging Face...")
    file_path = hf_hub_download(repo_id=repo_id, filename="openwebtext.parquet")
    df = pd.read_parquet(file_path)
    text_list = df['text'].head(num_samples).apply(lambda x: str(x)[:600]).tolist()
    print(f"Successfully loaded {len(text_list)} samples!")
    return text_list