import streamlit as st
import moviepy.editor as mp
import whisper
import os
import yt_dlp
import cv2
import mediapipe as mp_ai
from moviepy.video.fx.all import crop
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# Configuration de l'interface
st.set_page_config(page_title="Klap Clone Pro (Open Source)", layout="wide")
st.title("💎 Klap Clone Pro : Le Short Viral Parfait")
st.markdown("Personnalisez vos sous-titres et utilisez le **Face-Tracking IA** pour un rendu professionnel.")

# Initialisation de MediaPipe pour la détection de visage
mp_face_detection = mp_ai.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def get_face_center(frame):
    """Détecte le centre du visage le plus important dans une image."""
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        # On prend le premier visage détecté
        bbox = results.detections[0].location_data.relative_bounding_box
        center_x = bbox.xmin + (bbox.width / 2)
        return center_x
    return 0.5 # Par défaut au centre si aucun visage n'est trouvé

def download_youtube_video(url):
    output_path = os.path.join(TEMP_DIR, "input_video.mp4")
    if os.path.exists(output_path): os.remove(output_path)
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def process_video_pro(video_path, output_path, sub_color, sub_size, use_tracking):
    # 1. Transcription
    st.info("🎙️ Transcription IA en cours...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    segments = result['segments']

    # 2. Chargement Vidéo
    clip = mp.VideoFileClip(video_path)
    w, h = clip.size
    target_ratio = 9/16
    target_w = h * target_ratio

    # 3. Recadrage (Face-Tracking ou Centre)
    st.info("📐 Recadrage intelligent (9:16)...")
    if use_tracking:
        # Pour le prototype, on calcule le centre moyen sur les 5 premières secondes
        # Un vrai tracking dynamique par image est plus lourd mais possible.
        sample_frame = clip.get_frame(2) # On prend une image à 2 secondes
        x_center_rel = get_face_center(sample_frame)
        x_center = x_center_rel * w
    else:
        x_center = w / 2

    x1 = max(0, min(w - target_w, x_center - (target_w / 2)))
    x2 = x1 + target_w
    final_clip = crop(clip, x1=x1, y1=0, x2=x2, y2=h).resize(height=1280)

    # 4. Sous-titres Personnalisés
    st.info("✍️ Création des sous-titres stylisés...")
    subtitle_clips = []
    for segment in segments:
        if segment['start'] > 60: break # Limite à 60s pour la démo
        
        txt_clip = TextClip(
            segment['text'].strip().upper(), 
            fontsize=sub_size, 
            color=sub_color, 
            font='Arial-Bold', 
            stroke_color='black', 
            stroke_width=2,
            method='caption',
            size=(target_w*0.8, None)
        ).set_start(segment['start']).set_end(segment['end']).set_position(('center', 0.7), relative=True)
        
        subtitle_clips.append(txt_clip)

    # 5. Montage et Export
    result_video = CompositeVideoClip([final_clip.subclip(0, min(60, clip.duration))] + subtitle_clips)
    st.info("🎬 Rendu final...")
    result_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    return output_path

# --- Interface Latérale (Sidebar) ---
st.sidebar.header("🎨 Personnalisation")
sub_color = st.sidebar.color_picker("Couleur des sous-titres", "#FFFF00") # Jaune par défaut
sub_size = st.sidebar.slider("Taille de la police", 30, 100, 60)
use_tracking = st.sidebar.checkbox("Activer le Face-Tracking (IA)", value=True)

# --- Zone Principale ---
youtube_url = st.text_input("🔗 Lien YouTube :", placeholder="https://www.youtube.com/watch?v=...")

if youtube_url and st.button("🔥 Générer mon Short Pro"):
    try:
        video_path = download_youtube_video(youtube_url)
        output_path = os.path.join(TEMP_DIR, "short_pro.mp4")
        processed_video = process_video_pro(video_path, output_path, sub_color, sub_size, use_tracking)
        
        st.success("✅ Terminé !")
        st.video(processed_video)
        with open(processed_video, "rb") as f:
            st.download_button("⬇️ Télécharger le Short", f, file_name="short_viral_pro.mp4")
    except Exception as e:
        st.error(f"Erreur : {e}")
