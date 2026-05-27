from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import torch
import gc

def main():
    # 1. Initialize the model ONCE to save VRAM and time
    print("Loading model...")
    base_cfg = get_default_cfg()
    base_cfg["model_name"] = "gpt2-small"
    base_cfg['device'] = 'cuda'
    
    model = HookedTransformer.from_pretrained(base_cfg["model_name"]).to(base_cfg["dtype"]).to(base_cfg["device"])

    # Loop 1: JumpReLU comparison
    print("Starting JumpReLU loop...")
    for l1_coeff in [0.004, 0.0018, 0.0008]:
        cfg = get_default_cfg()
        cfg.update({
            "sae_type": "jumprelu",
            "model_name": "gpt2-small",
            "layer": 8,
            "site": "resid_pre",
            "dataset_path": "Skylion007/openwebtext",
            "aux_penalty": (1/32),
            "lr": 3e-4,
            "input_unit_norm": True,
            "top_k": 32,
            "dict_size": 768 * 16,
            "wandb_project": "batchtopk_comparison",
            "act_size": 768,
            "device": "cuda",
            "bandwidth": 0.001,
            "l1_coeff": l1_coeff
        })
        
        cfg = post_init_cfg(cfg)
        sae = JumpReLUSAE(cfg)
        activations_store = ActivationsStore(model, cfg)
        
        train_sae(sae, activations_store, model, cfg)
        clean_memory(sae, activations_store)

    # Loop 2: TopK & BatchTopK (varying top_k)
    print("Starting TopK & BatchTopK loop (varying top_k)...")
    for sae_type in ['topk', 'batchtopk']:
        for top_k in [16, 32, 64]:
            cfg = get_default_cfg()
            cfg.update({
                "sae_type": sae_type,
                "model_name": "gpt2-small",
                "layer": 8,
                "site": "resid_pre",
                "dataset_path": "Skylion007/openwebtext",
                "aux_penalty": (1/32),
                "lr": 3e-4,
                "input_unit_norm": True,
                "top_k": top_k,
                "dict_size": 768 * 16,
                "wandb_project": "batchtopk_comparison",
                "l1_coeff": 0.0,
                "act_size": 768,
                "device": "cuda",
                "bandwidth": 0.001
            })

            cfg = post_init_cfg(cfg)
            if cfg["sae_type"] == "topk":
                sae = TopKSAE(cfg)
            elif cfg["sae_type"] == "batchtopk":
                sae = BatchTopKSAE(cfg)

            activations_store = ActivationsStore(model, cfg)
            train_sae(sae, activations_store, model, cfg)
            clean_memory(sae, activations_store)

    # Loop 3: TopK & BatchTopK (varying dict_size)
    print("Starting TopK & BatchTopK loop (varying dict_size)...")
    for sae_type in ['topk', 'batchtopk']:
        for dict_size in [768*4, 768*8, 768*32]:
            cfg = get_default_cfg()
            cfg.update({
                "sae_type": sae_type,
                "model_name": "gpt2-small",
                "layer": 8,
                "site": "resid_pre",
                "dataset_path": "Skylion007/openwebtext",
                "aux_penalty": (1/32),
                "lr": 3e-4,
                "input_unit_norm": True,
                "top_k": 32,
                "dict_size": dict_size,
                "wandb_project": "batchtopk_comparison",
                "l1_coeff": 0.0,
                "act_size": 768,
                "device": "cuda",
                "bandwidth": 0.001
            })

            cfg = post_init_cfg(cfg)
            if cfg["sae_type"] == "topk":
                sae = TopKSAE(cfg)
            elif cfg["sae_type"] == "batchtopk":
                sae = BatchTopKSAE(cfg)

            activations_store = ActivationsStore(model, cfg)
            train_sae(sae, activations_store, model, cfg)
            clean_memory(sae, activations_store)

def clean_memory(sae, activations_store):
    """Helper to force clear VRAM between runs"""
    del sae
    del activations_store
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()