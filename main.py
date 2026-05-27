import sys
import torch
import gradio as gr
from datasets import load_dataset
from load_model import load_model, load_feature_titles, load_sample_dataset
from chat_functions import generate_sae_response
from sae_functions import sae_dashboard_analysis_2, generate_feature_titles, analyze_feature_globally_gradio

MODEL_NAME = "google/gemma-2-2b-it"
REPOSITORY = "beniaminbrad/yellow_goblin_gemma"
FOLDER = "folder" #In the HugFace Repo, the model and config files are to be put in a subfolder named "folder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TOKENS = 256
TEMP = 0.7
SYSTEM_PROMPT = "You are an honest and helpful AI Assistant"
IS_SAE = True
TOP_K_ACTIVATIONS = 10

SAMPLE_DATASET = load_sample_dataset(REPOSITORY, num_samples=5000)
print(f"Using device: {DEVICE}")
sae_model, sae = load_model(MODEL_NAME, REPOSITORY, FOLDER, DEVICE)
feature_titles = load_feature_titles(REPOSITORY)
print(sys.executable)
gr.close_all()

def get_steering_hook(sae, feature_id, coefficient, device):
    steering_vec = sae.W_dec[int(feature_id)].to(device)
    def hook_fn(resid_pre, hook):
        resid_pre = resid_pre + (coefficient * steering_vec)
        return resid_pre
    return hook_fn

def gradio_response(message, history, steer_enabled, steer_feat_id, steer_coeff):
    html = sae_dashboard_analysis_2(sae_model, sae, message, device=DEVICE, feature_title_dict=feature_titles, top_k=TOP_K_ACTIVATIONS) if IS_SAE else "<div style='font-family: monospace; font-size: 14px;'>SAE analysis disabled.</div>"
    
    if len(history) == 0:
        first_message = f"{SYSTEM_PROMPT}\n\n{message}"
        messages = [{"role": "user", "content": first_message}]
    else:
        messages = history + [{"role": "user", "content": message}]
        
    if steer_enabled and steer_feat_id is not None:
        hook_fn = get_steering_hook(sae, steer_feat_id, steer_coeff, DEVICE)
        target_layer = getattr(sae_model.cfg, 'layer', 8)
        hook_point = f'blocks.{target_layer}.hook_resid_pre'
        print(f"Injecting steering vector at hook: {hook_point}")
        with sae_model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
            response = generate_sae_response(sae_model, messages, NUM_TOKENS, TEMP)
    else:
        response = generate_sae_response(sae_model, messages, NUM_TOKENS, TEMP)
        
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, html

def show_loading_state():
    loading_html = "<div style='font-size: 16px; font-weight: bold; color: #ff9900; padding: 10px;'>⏳ Analyzing feature globally... Please wait.</div>"
    return gr.update(value=loading_html, visible=True), gr.update(visible=False), gr.update(visible=False)

def run_global_analysis(feature_id):
    if feature_id is None:
        return gr.update(value="<div style='color:red;'>Please enter a Feature ID.</div>"), gr.update(visible=False), gr.update(visible=False)
    
    fig, html = analyze_feature_globally_gradio(sae_model, sae, int(feature_id), SAMPLE_DATASET, DEVICE)
    
    if fig is None:
        return gr.update(value=html, visible=True), gr.update(visible=False), gr.update(visible=False)
        
    return gr.update(visible=False), gr.update(value=fig, visible=True), gr.update(value=html, visible=True)

with gr.Blocks(title="Gemma Chat") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type your message...")
            
            with gr.Accordion("Activation Addition (Steering)", open=False):
                gr.Markdown("Control model outputs by injecting SAE feature directions at inference time.")
                steer_enabled = gr.Checkbox(label="Enable Steering", value=False)
                with gr.Row():
                    steer_feat_id = gr.Number(label="Feature ID", precision=0)
                    steer_coeff = gr.Slider(minimum=-50.0, maximum=100.0, value=10.0, step=0.5, label="Injection Coefficient (c)")
            
        with gr.Column(scale=1):
            activation_display = gr.HTML(label="Top Activations")
            gr.Markdown("### Global Feature Analysis")
            with gr.Row():
                feat_input = gr.Number(label="Feature ID", precision=0, scale=3)
                analyze_btn = gr.Button("Analyze Globally", scale=1, variant="primary")
                
            global_loading_msg = gr.HTML(visible=False)
            global_plot = gr.Plot(visible=False)
            global_html = gr.HTML(visible=False)
            
    msg.submit(
        gradio_response, 
        inputs=[msg, chatbot, steer_enabled, steer_feat_id, steer_coeff], 
        outputs=[chatbot, activation_display]
    )

    analyze_btn.click(
        fn=show_loading_state, 
        inputs=[], 
        outputs=[global_loading_msg, global_plot, global_html]
    ).then(
        fn=run_global_analysis,
        inputs=[feat_input],
        outputs=[global_loading_msg, global_plot, global_html]
    )

demo.launch(
    server_name="127.0.0.1", 
    server_port=8080,      
    debug=True,              
    inline=False,            
    inbrowser=False,
    share=False, 
    prevent_thread_lock=True, 
)