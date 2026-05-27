import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from IPython.display import HTML, display # If using Jupyter/Colab
import numpy as np
   
def sae_dashboard_analysis_2(model, sae, query, device,feature_title_dict, top_k):
    top_k+=1
    print("Updated sae_dashboard_analysis_2 called with query:", query)
    # 1. Convert the input query to tokens and string tokens (for display)
    tokens = model.to_tokens(query).to(device)
    
    # Slice off the <bos> token (index 0) for display
    str_tokens = model.to_str_tokens(query)[1:] 

    hook_name = 'blocks.8.hook_resid_pre'
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

    hidden_states = cache[hook_name] 

    # 3. Pass the cached activations through the SAE to get feature activations
    with torch.no_grad():
        feature_acts = sae.encode(hidden_states)
    
    # Squeeze out the batch dimension AND slice off the <bos> token at index 0
    feature_acts = feature_acts.squeeze(0)[1:]

    # 4. Find the features that fired the hardest across this specific prompt
    max_acts_per_feature, _ = feature_acts.max(dim=0)
    
    # Get the indices of the top_k features
    top_feature_values, top_feature_indices = max_acts_per_feature.topk(top_k)

    # 5. Build the HTML output
    html_content = f"<h3 style='margin-bottom: 10px; color: #ddd;'>Top {top_k} Features Firing for this Prompt</h3>"

    for max_val, feature_idx in zip(top_feature_values, top_feature_indices):
        #Skip the first one since it's usually the same "most activating feature" that just captures overall activation magnitude rather than a specific interpretable feature
        if feature_idx == top_feature_indices[0]:
            continue
        feature_idx = feature_idx.item()
        max_val = max_val.item()
        
        if max_val == 0:
            continue

        feature_title = feature_title_dict.get(feature_idx, "Unknown")
        html_content += f"<div style='margin-top: 15px; color: #ddd;'><b>Feature #{feature_idx}: \"{feature_title}\"</b> (Absolute Max Act: {max_val:.2f})</div>"
        # [FIX 1]: Added word-wrap: break-word to ensure the container itself allows wrapping
        html_content += "<div style='line-height: 2.2em; padding: 15px; background: #f9f9f9; color: #000; border-radius: 5px; border: 1px solid #ddd; font-family: monospace; font-size: 14px; margin-top: 5px; word-wrap: break-word;'>"
        
        # Get the activations for THIS specific feature across all remaining tokens
        this_feature_acts = feature_acts[:, feature_idx]

        # Build the heatmap for this feature
        for token, act in zip(str_tokens, this_feature_acts):
            val = act.item()
            
            alpha = val / max_val if max_val > 0 else 0
            alpha = min(1.0, alpha) 
            
            color = f"rgba(255, 150, 0, {alpha})"
            
            # [FIX 2]: Added a replacement for '\n' so multi-line prompts don't break the formatting
            clean_token = token.replace(' ', '&nbsp;').replace('<bos>', '&lt;bos&gt;').replace('\n', '↵')
            if not clean_token.strip() and '&nbsp;' not in clean_token and '↵' not in clean_token:
                clean_token = "&nbsp;" * max(1, len(clean_token))
            
            # [FIX 3]: Added `display: inline-block;` and vertical margin (`margin: 2px 1px;`)
            html_content += f"<span style='display: inline-block; background-color: {color}; padding: 2px 0px; border-radius: 2px; margin: 2px 1px;'>{clean_token}</span>"
            
        html_content += "</div>"

    if html_content.endswith("</h3>"):
        html_content += "<div style='color: #888;'>No features activated for this prompt.</div>"

    return html_content

def generate_feature_titles(model, sae, device):
    print("Pre-computing feature titles via Logit Lens...")
    feature_titles = {}
    
    # Get the number of features from your SAE
    num_features = sae.W_dec.shape[0]
    
    with torch.no_grad():
        for feat_idx in tqdm(range(num_features)):
            # 1. Get the direction vector and cast it to the model's dtype (bfloat16)
            feature_dir = sae.W_dec[feat_idx].to(device).to(model.W_U.dtype)
            
            # 2. Multiply against Gemma's unembedding matrix (W_U)
            logits = feature_dir @ model.W_U 
            
            # 3. Get the top 1 token index
            top_token_idx = logits.argmax(dim=-1).item()
            
            # 4. Convert token index to a readable string
            top_word = model.to_string(top_token_idx).strip()
            
            # Clean up empty strings or newlines
            if not top_word or top_word == '\n':
                top_word = "[Whitespace/Formatting]"
                
            feature_titles[feat_idx] = top_word

    return feature_titles

def analyze_feature_globally_gradio(model, sae, feature_idx, dataset_texts, device, top_contexts_k=5):
    # --- 1. LOGIT LENS ---
    feature_dir = sae.W_dec[feature_idx].to(device).to(model.W_U.dtype)
    logits = feature_dir @ model.W_U
    top_token_ids = logits.topk(5)[1]
    top_words = [model.to_string(t.item()).strip() for t in top_token_ids]
    
    # --- 2. DATA GATHERING ---
    all_activations = []
    context_records = []
    hook_name = 'blocks.8.hook_resid_pre'

    import torch
    with torch.no_grad():
        for i, text in enumerate(dataset_texts):
            print(f"Processing text {i+1}/{len(dataset_texts)}", end='\r')
            tokens = model.to_tokens(text).to(device)
            str_tokens = model.to_str_tokens(text)[1:] 
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            feature_acts = sae.encode(cache[hook_name])
            target_acts = feature_acts.squeeze(0)[1:, feature_idx].cpu().numpy()
            max_act = target_acts.max()
            
            if max_act > 0:
                all_activations.extend(target_acts[target_acts > 0])
                context_records.append({'text_idx': i, 'str_tokens': str_tokens, 'activations': target_acts, 'max_act': max_act})

    if not context_records:
        return None, f"<div style='color:red;'>⚠️ Feature #{feature_idx} did not activate in this dataset.</div>"

    # --- 3. CREATE MATPLOTLIB FIGURE ---
    fig = plt.figure(figsize=(10, 4))
    plt.hist(all_activations, bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.title(f"Feature #{feature_idx} Activation Histogram")
    plt.xlabel("Activation Strength")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout() # Ensures it fits well in Gradio

    # --- 4. BUILD HTML ---
    context_records.sort(key=lambda x: x['max_act'], reverse=True)
    
    html_content = f"<h3>Top Predictions: {', '.join(top_words)}</h3>"
    
    for rank, record in enumerate(context_records[:top_contexts_k]):
        max_val = record['max_act']
        html_content += f"<div style='margin-top: 10px; color: #555;'><b>Rank {rank+1}</b> (Max Act: {max_val:.2f})</div>"
        html_content += "<div style='line-height: 2.2em; padding: 10px; background: #f9f9f9; color: #000; border-radius: 5px; border: 1px solid #ddd; word-wrap: break-word;'>"
        for token, act in zip(record['str_tokens'], record['activations']):
            alpha = min(1.0, act / max_val) if max_val > 0 else 0
            color = f"rgba(255, 150, 0, {alpha})"
            clean_token = token.replace(' ', '&nbsp;').replace('<bos>', '&lt;bos&gt;').replace('\n', '↵')
            if not clean_token.strip(): clean_token = "&nbsp;"
            html_content += f"<span style='background-color: {color}; margin: 1px;'>{clean_token}</span>"
        html_content += "</div>"

    return fig, html_content

def get_steering_hook(sae, feature_id, coefficient, device):
    steering_vec = sae.W_dec[int(feature_id)].to(device)
    steering_vec = steering_vec / (steering_vec.norm() + 1e-8) #normalized
    
    def hook_fn(resid_pre, hook):
        return resid_pre + (coefficient * steering_vec)
        
    return hook_fn