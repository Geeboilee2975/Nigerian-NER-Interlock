import os
import time
import unicodedata
import streamlit as st
import PyPDF2
import pandas as pd
import trafilatura  
import speech_recognition as sr  
from gtts import gTTS 
import base64
import io  
import tempfile  # NEW: For robust disk-spooling
from pydub import AudioSegment  # NEW: For audio transcoding
from streamlit_mic_recorder import mic_recorder 
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. PAGE SETUP
st.set_page_config(page_title="Master Nigerian NER Interlock", page_icon="🇳🇬", layout="wide")
st.title("🇳🇬 Master Nigerian NER System")
st.markdown("### **Electronic & Computer Engineering - Neural Phonetic Interface**")

# 2. MODEL LOADING
@st.cache_resource 
def load_nigerian_model():
    try:
        repo_id = "gbolahan219/Nigerian-NER-Model" 
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForTokenClassification.from_pretrained(repo_id)
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except Exception as e:
        st.error(f"❌ System Load Error: {e}")
        return None

nlp_pipe = load_nigerian_model()

# --- 3. NEURAL VOICE ENGINE ---
def speak_results_neural(text):
    try:
        tts = gTTS(text=text, lang='en', tld='com.ng')
        tts.save("temp_speech.mp3")
        with open("temp_speech.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
        time.sleep(2) 
        os.remove("temp_speech.mp3")
    except Exception as e:
        st.warning(f"Audio Synthesis Warning: {e}")

# 4. ENTITY CONFIGURATION
entity_config = {
    "PER": {"name": "PERSON", "color": "#FF4B4B"}, "GPE": {"name": "GPE (STATE/CITY)", "color": "#7D3C98"},
    "ORG": {"name": "ORGANIZATION", "color": "#FFD700"}, "DATE": {"name": "DATE/TIME", "color": "#28A745"},
    "CARDINAL": {"name": "CARDINAL", "color": "#00CED1"}, "EVENT": {"name": "EVENT", "color": "#FF8C00"},
    "FAC": {"name": "FACILITY", "color": "#4682B4"}, "LANGUAGE": {"name": "LANGUAGE", "color": "#6A5ACD"},
    "LAW": {"name": "LAW/LEGAL", "color": "#8B4513"}, "ORDINAL": {"name": "ORDINAL", "color": "#D2691E"},
    "PERCENT": {"name": "PERCENTAGE", "color": "#FF1493"}, "PRODUCT": {"name": "PRODUCT", "color": "#008080"},
    "QUANTITY": {"name": "QUANTITY", "color": "#708090"}, "WORK_OF_ART": {"name": "WORK OF ART", "color": "#DC143C"},
    "LOC": {"name": "LOCATION", "color": "#1C83E1"}, "MONEY": {"name": "CURRENCY", "color": "#32CD32"},
    "TIME": {"name": "TIME", "color": "#ADFF2F"}, "NORP": {"name": "NORP (GROUP)", "color": "#DEB887"}
}

label_map = {
    "LABEL_1": "PER", "LABEL_2": "PER", "LABEL_3": "ORG", "LABEL_4": "ORG",
    "LABEL_5": "GPE", "LABEL_6": "GPE", "LABEL_7": "DATE", "LABEL_8": "CARDINAL",
    "LABEL_9": "EVENT", "LABEL_10": "FAC", "LABEL_11": "LANGUAGE", "LABEL_12": "LAW",
    "LABEL_13": "ORDINAL", "LABEL_14": "PERCENT", "LABEL_15": "PRODUCT",
    "LABEL_16": "QUANTITY", "LABEL_17": "WORK_OF_ART", "LABEL_18": "LOC"
}

# 5. SIDEBAR
st.sidebar.header("🕹️ System Controls")
upload_method = st.sidebar.radio("Select Input Source:", ["Voice Command", "Web URL Scraper", "Upload PDF Document", "Manual/Language Sections"])

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
threshold = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 60) / 100
enable_voice_feedback = st.sidebar.checkbox("Enable Neural Read-Out", value=True)

if 'voice_data' not in st.session_state: st.session_state['voice_data'] = ""
final_input = ""

# --- 6. INPUT AREA (ROBUST TRANSCODING) ---
st.divider()
if upload_method == "Voice Command":
    st.subheader("🎤 Voice-to-Interlock Ingestion")
    st.info("Click to Record and speak clearly into your microphone.")
    
    audio_file = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop & Process",
        key='web_recorder'
    )

    if audio_file:
        try:
            # 1. Create a temporary disk buffer for the incoming signal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file['bytes'])
                temp_audio_path = temp_audio.name

            # 2. Transcode bitstream to processable WAV via FFmpeg
            audio_segment = AudioSegment.from_file(temp_audio_path)
            audio_segment.export(temp_audio_path, format="wav")
            
            # 3. Use SpeechRecognition on the stabilized signal
            r = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = r.record(source)
                st.session_state['voice_data'] = r.recognize_google(audio_data)
                st.success("✅ Voice Signal Stabilized & Captured!")
            
            # 4. Clean up temporary storage (Engineering discipline)
            os.remove(temp_audio_path)
            
        except Exception as e:
            st.error(f"Signal Transcoding Error: {e}")
            st.info("Ensure packages.txt contains 'ffmpeg' and reboot the app if error persists.")
                
    if st.session_state['voice_data']:
        final_input = st.text_area("Recognized Speech Signal:", value=st.session_state['voice_data'])

elif upload_method == "Web URL Scraper":
    st.subheader("🌐 Web Intelligence Ingestion")
    url_link = st.text_input("Enter URL:")
    if url_link:
        downloaded = trafilatura.fetch_url(url_link)
        if downloaded:
            raw_text = trafilatura.extract(downloaded)
            final_input = unicodedata.normalize('NFC', raw_text)

elif upload_method == "Upload PDF Document":
    st.subheader("📁 PDF Document Analysis")
    uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        raw_text = "".join([page.extract_text() for page in pdf_reader.pages])
        final_input = unicodedata.normalize('NFC', raw_text)
        st.info(f"📄 Document Loaded: {uploaded_file.name}")

else:
    st.subheader("✍️ Manual Text Input")
    lang_section = st.selectbox("Language Section:", ["Manual Input", "Nigerian Pidgin", "Yoruba", "Igbo", "Hausa"])
    
    samples = {
        "Nigerian Pidgin": "Obi buy 50 bags of rice for Lagos last week inside Nigeria.",
        "Yoruba": "Ojo kọrin 'Ojumo Re' ni Eko ninu oṣu kejila.",
        "Igbo": "Buhari bịanyere aka na Law 10 n'ime Abuja.",
        "Hausa": "Gwamna ya ba da kashi 20 cikin dari na kudin a Kano."
    }
    
    sample_val = samples.get(lang_section, "") if lang_section != "Manual Input" else ""
    final_input = st.text_area("Input Text Editor:", value=sample_val, height=250)
    final_input = unicodedata.normalize('NFC', final_input)

# --- 7. CONDITIONAL PREVIEW LAYER ---
if final_input and upload_method in ["Web URL Scraper", "Upload PDF Document"]:
    st.markdown("### 📝 Document Preview")
    final_input = st.text_area("Final Signal Verification:", value=final_input, height=200)

# --- 8. ANALYSIS & EXECUTION ---
if final_input:
    if st.button("🚀 EXECUTE AI INTERLOCK"):
        if nlp_pipe:
            start_time = time.time()
            results = nlp_pipe(final_input)
            latency = time.time() - start_time
            valid_results = [res for res in results if label_map.get(res['entity_group'], "O") != "O" and res['score'] >= threshold]
            
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Validated Entities", len(valid_results))
            m2.metric("Latency", f"{latency:.3f}s")
            avg_conf = (sum([r['score'] for r in valid_results])/len(valid_results)) if valid_results else 0
            m3.metric("Signal Strength", f"{int(avg_conf*100)}%")
            m4.metric("Status", "HIGH (Logic 1)" if valid_results else "LOW (Logic 0)")

            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.write("🔍 **Entity Grid View:**")
                grid_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
                for entity in valid_results:
                    key = label_map.get(entity['entity_group'], "O")
                    cfg = entity_config.get(key, {"name": key, "color": "#888888"})
                    grid_html += f'<div style="background-color:{cfg["color"]}; color:white; padding:5px 12px; border-radius:15px; font-weight:bold; border:1px solid white; font-size: 0.8rem;">{cfg["name"]}: {entity["word"]}</div>'
                st.markdown(grid_html + '</div>', unsafe_allow_html=True)
                
                st.markdown("<br><br>", unsafe_allow_html=True) 
                st.divider()
                
                if valid_results:
                    st.write("📊 **Entity Distribution Frequency:**")
                    type_counts = pd.Series([label_map.get(res['entity_group']) for res in valid_results]).value_counts()
                    st.bar_chart(type_counts)

            with col_r:
                if valid_results:
                    st.write("📈 **Confidence Signal Oscilloscope:**")
                    st.line_chart([r['score'] for r in valid_results])
                    df = pd.DataFrame([{"Type": label_map.get(e['entity_group'], "O"), "Text": e['word'], "Conf": f"{e['score']:.1%}"} for e in valid_results])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            if enable_voice_feedback:
                count = len(valid_results)
                if count > 0:
                    interpretation = f"Interlock protocol engaged. Found {count} validated entities. "
                    grouped_data = {}
                    for e in valid_results:
                        cat = entity_config.get(label_map.get(e['entity_group'], "O"), {}).get("name", "Unknown")
                        if cat not in grouped_data: grouped_data[cat] = []
                        grouped_data[cat].append(e['word'])
                    for cat, names in grouped_data.items():
                        unique_names = list(dict.fromkeys(names))
                        interpretation += f"For {cat}, I found: {', '.join(unique_names)}. "
                    interpretation += "All telemetry processed. Status high."
                else:
                    interpretation = "Analysis complete. No entities detected. Status low."
                speak_results_neural(interpretation)

# 9. SIDEBAR LEGEND
st.sidebar.divider()
for key, val in entity_config.items():
    st.sidebar.markdown(f'<span style="color:{val["color"]}">●</span> {val["name"]}', unsafe_allow_html=True)