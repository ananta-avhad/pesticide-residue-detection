import streamlit as st
import sys
import os
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import json
import xml.etree.ElementTree as ET
from dicttoxml import dicttoxml
import tempfile 

# --- Fixed import conflict by aliasing Twilio's Client ---
from twilio.rest import Client as TwilioClient
from supabase import create_client, Client

# --- Securely loading secrets using Streamlit Secrets ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except KeyError:
    st.error("Missing Supabase credentials. Please set them in your Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# for DB insertion
def save_user_phone(phone_number):
    try:
        data = {
            "phone": phone_number
        }
        supabase.table("users").insert(data).execute()
        return True, None
    except Exception as e:
        return False, str(e)

# test DB insert function
def test_insert():
    # You can change this dummy number if needed
    return save_user_phone("+911234567890")

# for sms sending
def send_sms(phone, message):
    try:
        ACCOUNT_SID = st.secrets.get("TWILIO_SID", os.getenv("TWILIO_SID"))
        AUTH_TOKEN = st.secrets.get("TWILIO_AUTH", os.getenv("TWILIO_AUTH"))
        TWILIO_NUMBER = st.secrets.get("TWILIO_NUMBER", os.getenv("TWILIO_NUMBER"))

        # Using the aliased TwilioClient
        client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)

        message = client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=phone
        )

        return True, message.sid

    except Exception as e:
        return False, str(e)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import PesticidePredictor

# Page configuration
st.set_page_config(
    page_title="Pesticide Residue Detection",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .safe {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .contaminated {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .uncertain {
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
        color: #333333; 
    }
    .uncertain h3, .uncertain p {
        color: #333333; 
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (Model Loading)
if 'predictor' not in st.session_state:
    try:
        # Load the predictor class which loads the trained model
        st.session_state.predictor = PesticidePredictor()
        st.session_state.model_loaded = True
        st.session_state.error_message = None
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.error_message = f"Failed to load model. Ensure 'final_model.h5' exists and check config.yaml. Error: {str(e)}"

# Header
st.markdown('<h1 class="main-header">🥗 Pesticide Residue Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image of fruits or vegetables to detect pesticide residues</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    
    # Optional test button
    if st.button("Test DB Insert (Dummy)"):
        success, error = test_insert()
        if success:
            st.success("✅ Dummy Insert ho gaya!")
        else:
            st.error("❌ Insert fail hua")
            st.code(error)
    
    st.sidebar.header("📱 Phone & Alerts")
    
    # --- FIX 1: Single unified input field for phone number ---
    phone = st.sidebar.text_input("Enter phone (+91...)")

    col1, col2 = st.columns(2)
    
    # Register Button
    with col1:
        if st.button("Register DB"):
            if phone:
                success, error = save_user_phone(phone)
                if success:
                    st.success("✅ Saved!")
                else:
                    st.error("❌ Failed")
                    st.code(error)
            else:
                st.warning("⚠️ Enter phone")

    # SMS Button
    with col2:
        if st.button("Send SMS"):
            # --- FIX 2: Added validation so it doesn't run if empty ---
            if phone:
                success, res = send_sms(phone, "🔥 Test SMS from your ML app!")
                if success:
                    st.success("✅ SMS sent!")
                else:
                    st.error("❌ Failed")
                    st.code(res)
            else:
                st.warning("⚠️ Enter phone")
    
    st.header("ℹ️ About")
    st.write("""
    This application uses machine learning to detect pesticide residues on fruits and vegetables.
    
    **How to use:**
    1. Upload an image
    2. Wait for analysis
    3. View results and download report
    """)
    
    st.header("⚙️ Settings")
    show_probabilities = st.checkbox("Show probability breakdown", value=True)
    auto_download = st.checkbox("Auto-download reports", value=False)
    
    st.header("📊 Model Info")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
        st.info("Validation Accuracy: Check `training_history.png` in models/saved_models/")
    else:
        st.error("❌ Model not loaded")
        if st.session_state.error_message:
            st.code(st.session_state.error_message, language='python')

# Main content
if not st.session_state.model_loaded:
    st.error("⚠️ Model not loaded. Please train the model first using train.py and ensure 'final_model.h5' is in models/saved_models/.")
    st.info("Run: `python src/train.py` to train the model")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    help="Limit 200MB per file. Upload an image of fruits or vegetables"
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if not file_extension:
            file_extension = ".jpg" 

        temp_path = os.path.join(tmpdir, "uploaded_image" + file_extension)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(temp_path)
            st.image(image, caption=uploaded_file.name) 
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing image..."):
                try:
                    result = st.session_state.predictor.process_image(temp_path)
                    
                    is_contaminated = result.get('is_contaminated', False)
                    confidence = result.get('confidence', 0.0)
                    meets_threshold = result.get('meets_confidence_threshold', False)
                    
                    if is_contaminated:
                        if meets_threshold:
                            result_class = "contaminated"
                            icon = "⚠️"
                            status = "HIGH RISK OF CONTAMINATION"
                            color = "#F44336"
                        else:
                            result_class = "uncertain"
                            icon = "❓"
                            status = "POSSIBLY CONTAMINATED (LOW CONFIDENCE)"
                            color = "#FF9800"
                    else:
                        if meets_threshold:
                            result_class = "safe"
                            icon = "✅"
                            status = "SAFE (NO CONTAMINATION DETECTED)"
                            color = "#4CAF50"
                        else:
                            result_class = "uncertain"
                            icon = "❓"
                            status = "UNCERTAIN RESULT"
                            color = "#FF9800"
                            
                    st.markdown(f"""
                    <div class="result-box {result_class}">
                        <h2 style="color: {color};">{icon} {status}</h2>
                        <h3>Confidence: {confidence:.1%}</h3>
                        <p><strong>Predicted Class:</strong> {result['predicted_class']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if show_probabilities:
                        st.subheader("📊 Probability Breakdown")
                        prob_data = result['probabilities']
                        
                        for class_name, prob in prob_data.items():
                            st.progress(min(1.0, max(0.0, float(prob))), text=f"**{class_name.capitalize()}** probability: **{prob:.1%}**")
                    
                    st.subheader("💡 Recommendations")
                    if is_contaminated and meets_threshold:
                        st.warning("⚠️ High contamination risk detected. **Do not consume.** Further expert testing recommended.")
                    elif is_contaminated and not meets_threshold:
                        st.info("Possible contamination detected with low confidence. **Consider rinsing thoroughly and re-testing.**")
                    elif not is_contaminated and meets_threshold:
                        st.success("✅ No contamination detected. Product appears safe.")
                    else: 
                        st.info("Result uncertain. **Consider retesting or verifying source.**")
                    
                    st.subheader("📥 Download Reports")
                    
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_base = f"{base_name}_{timestamp}"

                    col_txt, col_json, col_xml = st.columns(3)
                    
                    report_data = {
                        "filename": uploaded_file.name,
                        "timestamp": timestamp,
                        "prediction": result['predicted_class'],
                        "confidence_score": confidence,
                        "status": status,
                        "probabilities": result['probabilities']
                    }

                    txt_content = st.session_state.predictor.generate_txt_report(result)
                    with col_txt:
                        st.download_button(
                            label="📄 TXT Report",
                            data=txt_content,
                            file_name=f"{output_base}_report.txt",
                            mime="text/plain"
                        )
                    
                    json_content = st.session_state.predictor.generate_json_report(result)
                    with col_json:
                        st.download_button(
                            label="📊 JSON Report",
                            data=json_content,
                            file_name=f"{output_base}_report.json",
                            mime="application/json"
                        )
                    
                    xml_content = st.session_state.predictor.generate_xml_report(result)
                    with col_xml:
                        st.download_button(
                            label="📋 XML Report",
                            data=xml_content,
                            file_name=f"{output_base}_report.xml",
                            mime="application/xml"
                        )
                    
                except Exception as e:
                    st.error(f"Error during prediction or report generation. Please check the `src/predict.py` file.")
                    st.caption("Detailed error for debugging:")
                    st.code(f"{type(e).__name__}: {e}", language='python')
                    
else:
    st.info("👆 Please upload an image to begin analysis")
    
    st.subheader("📝 Supported Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("✅ **Image Upload**")
        st.write("Supports JPG, PNG, TIFF")
    
    with col2:
        st.write("🎯 **Real-time Detection**")
        st.write("Instant analysis results")
    
    with col3:
        st.write("📊 **Detailed Reports**")
        st.write("TXT, XML, JSON formats")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Pesticide Residue Detection System v1.0</strong></p>
</div>
""", unsafe_allow_html=True)