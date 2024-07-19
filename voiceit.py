from threading import Thread
from pathlib import Path
import gradio as gr
import subprocess
import shutil
import time
import copy
import glob
import json
import os

CURRENT_DIR = Path(__file__).resolve().parent
MODELOS = None
for dir_name in ["modelos", "models"]:
    potential_path = CURRENT_DIR / dir_name
    if potential_path.exists():
        MODELOS = potential_path
        break

if MODELOS is None:
    raise FileNotFoundError("No se han encontrado los directorios 'modelos' ni 'models' en el directorio actual.")

INFERENCE_OUTPUT_DIRNAME = CURRENT_DIR / "inference_output"

def get_container_format(filename):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=format_name", "-of", "default=noprint_wrappers=1:nokey=1", filename]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        raise ValueError(f"Error: {error.decode()}")
    return output.decode().strip()

def cleanup_dirs():
    dir_path = Path(INFERENCE_OUTPUT_DIRNAME)
    if dir_path.exists():
        shutil.rmtree(dir_path)

def get_speakers():
    global speakers
    speakers = []
    for _, dirs, _ in os.walk(MODELOS):
        for folder in dirs:
            cur_speaker = {}
            g = glob.glob(os.path.join(MODELOS, folder, 'G_*.pth'))
            if not len(g):
                continue
            cur_speaker["model_path"] = g[0]
            cur_speaker["model_folder"] = folder
            cur_speaker["cluster_path"] = ""

            cfg = glob.glob(os.path.join(MODELOS, folder, '*.json'))
            if not len(cfg):
                continue
            cur_speaker["cfg_path"] = cfg[0]
            with open(cur_speaker["cfg_path"]) as f:
                try:
                    cfg_json = json.loads(f.read())
                except Exception as e:
                    print("Archivo json malformado en" + folder)
                for name, i in cfg_json["spk"].items():
                    cur_speaker["name"] = name
                    cur_speaker["id"] = i
                    if not name.startswith('.'):
                        speakers.append(copy.copy(cur_speaker))
    return sorted(speakers, key=lambda x: x["name"].lower())

def run_inference(speaker, path, f0_method, transpose, noise_scale, cluster_ratio):
    model_path = speaker["model_path"]
    config_path = speaker["cfg_path"]
    cluster_path = speaker["cluster_path"]
    cluster_args = f"-k {cluster_path} -r {cluster_ratio}" if cluster_path and cluster_ratio > 0 else ""
    inference_cmd = f"svc infer {path.absolute()} -m {model_path} -c {config_path} {cluster_args} -t {transpose} --f0-method crepe -n 0.4 -o {INFERENCE_OUTPUT_DIRNAME}/{path.name} --no-auto-predict-f0"
    result = subprocess.run(inference_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stderr:
        if "AttributeError" in result.stderr:
            return  None, gr.Textbox.update("⚠️ Modelo SVC incompatible.")
    if not list(Path(INFERENCE_OUTPUT_DIRNAME).glob("*")):
        return  None, gr.Textbox.update("⚠️ Error.")

def convert(speaker_box, audio):
    speaker = next((x for x in speakers if x["name"] == speaker_box), None)
    if not speaker:
        return None, gr.Textbox.update("⚠️ Selecciona un modelo.")
    if not audio:
        return None, gr.Textbox.update("⚠️ Sube un audio.")
    file_path = os.path.join(os.getcwd(), str(audio))
    transpose = 0
    cluster_ratio = 0
    if os.path.exists(INFERENCE_OUTPUT_DIRNAME):
        cleanup_dirs()
    os.makedirs("inference_output", exist_ok=True)
    ts0 = time.time()
    run_inference(speaker, Path(file_path), 0, 0, 0.4, 0)
    final_filename = f"output{Path(file_path).suffix}"
    shutil.move(Path(INFERENCE_OUTPUT_DIRNAME, Path(file_path).name), Path(final_filename))
    cleanup_dirs()
    os.remove(file_path)
    ts1 = time.time()
    tiempo1 = int(ts1 - ts0)
    return final_filename, gr.Textbox.update("👌 ¡Voz cambiada!", label=f"Tiempo total: {tiempo1} segundos")

def clear():
    shutil.rmtree(INFERENCE_OUTPUT_DIRNAME, ignore_errors=True)
    tmp_files = glob.glob("*.tmp")
    for f in tmp_files:
        os.remove(f)
    return gr.Dropdown.update(value="Elige un modelo de voz"), None, gr.Textbox.update("🗑️ Datos borrados.", label=f"Información")


css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
footer {
    visibility: hidden;
    display: none;
}
.center-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
"""

with gr.Blocks(
    css=css,
    title="VoiceIt! - Pavloh",
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        radius_size="lg",
        text_size="lg"
    ).set(loader_color="#0B0F19", shadow_drop='*shadow_drop_lg', block_border_width="3px")
) as pavloh:
    gr.HTML("""
        <div class="center-container">
            <div style="display: flex; justify-content: center;">
                <a href="https://github.com/ImPavloh/voiceit/blob/main/LICENSE" target="_blank">
                    <img src="https://img.shields.io/github/license/impavloh/voiceit?style=for-the-badge&logo=github&logoColor=white" alt="Licencia">
                </a>
                <a href="https://github.com/impavloh/voiceit" target="_blank">
                    <img src="https://img.shields.io/badge/repositorio-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
                </a>
                <form action="https://www.paypal.com/donate" method="post" target="_blank">
                    <input type="hidden" name="hosted_button_id" value="6FPWP9AWEKSWJ" />
                    <input type="image" src="https://img.shields.io/badge/apoyar-%2300457C.svg?style=for-the-badge&logo=paypal&logoColor=white" border="0" name="submit" alt="Botón donar con PayPal" />
                    <img alt="" border="0" src="https://www.paypal.com/es_ES/i/scr/pixel.gif" width="1" height="1" />
                </form>
                <a href="https://twitter.com/impavloh" target="_blank">
                    <img src="https://img.shields.io/badge/Seguir-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
                </a>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;">
                <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px">🗣️ VoiceIt! - Un proyecto de <a style="text-decoration: underline;" href="https://twitter.com/impavloh">Pavloh</a></h1>
            </div>
            <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">Cambia la voz de audios utilizando modelos pre-entrenados de streamers.</p>
        </div>
    """)
    with gr.Row(elem_id="1",equal_height=True):
        with gr.Column():
            d1 = gr.Dropdown([x["name"] for x in get_speakers()], label="📦 Selecciona un modelo", value="Elige un modelo de voz")
            audio = gr.Audio(label="🗣️ Sube un audio", type="filepath")
        with gr.Column():
            a2 = gr.Audio(label="🔊 Resultado", type="filepath")
            t1 = gr.Textbox(type="text", label="📄 Información", value="Elige un modelo y un audio para cambiar la voz.")

    with gr.Row():
        b0 = gr.Button("🗑️ Borrar")
        b1 = gr.Button("🎤 Cambiar voz", variant="primary")
        b0.click(clear, outputs=[d1, audio, t1])
        b1.click(convert, inputs=[d1, audio], outputs=[a2, t1])

    with gr.Row():
        with gr.Accordion(label="Información importante", open=False):
            gr.HTML("""
                <center>
                    <p>
                        <i> Ten en cuenta que los audios deben contener solamente una voz y estar libres de ruido o música de fondo. </i>
                    </p>
                    <p>
                        <i> Asegúrate de que el nombre del archivo no contenga espacios ni símbolos raros, utilizando solo caracteres alfanuméricos y guiones bajos (_) para separar palabras si es necesario. </i>
                    </p>
                    <p>
                        <i> Al utilizar este sitio web, aceptas la <a style="text-decoration: underline;" href="https://github.com/ImPavloh/voiceit/blob/main/LICENSE">licencia</a> y <a style="text-decoration: underline;" href="https://github.com/ImPavloh/voiceit/blob/main/TERMINOS_DE_USO.txt">condiciones de uso</a>. </i>
                    </p>
                </center>
            """)

if __name__ == "__main__":
    pavloh.launch()
