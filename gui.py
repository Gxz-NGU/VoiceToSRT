import gradio as gr
import os
import shutil
from backend import VoiceAligner

# Global aligner instance to avoid reloading model
aligner = None
aligner_model = None

def get_aligner(model_name="base"):
    global aligner, aligner_model
    if aligner is None or aligner_model != model_name:
        aligner = VoiceAligner(model_name=model_name)
        aligner_model = model_name
    return aligner

def process_alignment(audio_file, text_file, language, model_name):
    """
    Handler for Gradio interface.
    """
    if audio_file is None or text_file is None:
        return None, "Please upload both audio and text files."

    try:
        # Determine output path
        output_dir = os.path.dirname(audio_file)
        # Use audio filename but change extension to .srt
        audio_filename = os.path.basename(audio_file)
        base_name = os.path.splitext(audio_filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.srt")
        
        # Determine language code
        lang_code = None
        if language == "Korean (ko)":
            lang_code = "ko"
        elif language == "Japanese (ja)":
            lang_code = "ja"
        elif language == "Chinese (zh)":
            lang_code = "zh"
        elif language == "English (en)":
            lang_code = "en"
        
        # Run alignment
        # Initialize model if not already done
        model = get_aligner(model_name=model_name)
        
        # Status update
        print(f"Processing: {audio_filename} with text {os.path.basename(text_file)}...")
        
        model.align_transcript(
            audio_path=audio_file,
            text_path=text_file,
            output_srt_path=output_path,
            language=lang_code
        )
        
        return output_path, f"Success! SRT generated at {output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Define the interface
with gr.Blocks(title="VoiceToSRT - Precise Alignment") as app:
    gr.Markdown("# VoiceToSRT: Audio & Text Alignment Tool")
    gr.Markdown("Upload your audio file and the corresponding text transcript to generate a precisely aligned SRT subtitle file.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Audio File", type="filepath", sources=["upload"])
            text_input = gr.File(label="Transcript Text File (.txt)", file_types=[".txt"], type="filepath")
            language_input = gr.Dropdown(
                choices=["Auto", "Korean (ko)", "Japanese (ja)", "Chinese (zh)", "English (en)"], 
                value="Auto", 
                label="Language"
            )
            model_input = gr.Dropdown(
                choices=["base", "small", "medium", "large-v2", "large-v3"],
                value="base",
                label="Model"
            )
            submit_btn = gr.Button("Generate SRT", variant="primary")
        
        with gr.Column():
            output_file = gr.File(label="Download SRT")
            status_text = gr.Textbox(label="Status", interactive=False)
    
    submit_btn.click(
        fn=process_alignment,
        inputs=[audio_input, text_input, language_input, model_input],
        outputs=[output_file, status_text]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
