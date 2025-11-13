import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.persona_loader import PERSONA_DISPLAY_NAMES, list_personas
from src.simple_voice_handler import SimpleVoiceHandler

# Configuration
APP_TITLE = "AI Voice Assistant"
BASE_DIR = Path(__file__).parent


def load_personas():
    """Load available personas with display names for the UI."""
    items = []
    for key, data in list_personas(BASE_DIR):
        display_name = PERSONA_DISPLAY_NAMES.get(key, data.get("name", key))
        items.append((display_name, data))
    return items


def init_session_state():
    """Initialize session state"""
    if "voice_handler" not in st.session_state:
        st.session_state.voice_handler = None
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "status" not in st.session_state:
        st.session_state.status = "Ready"
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "current_user_text" not in st.session_state:
        st.session_state.current_user_text = ""
    if "current_assistant_text" not in st.session_state:
        st.session_state.current_assistant_text = ""
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = None


def create_callbacks():
    """Create callbacks for voice handler"""
    def on_status(status):
        st.session_state.status = status
    
    def on_user_text(text):
        st.session_state.current_user_text = text
        st.session_state.conversation.append({"role": "user", "text": text})
    
    def on_assistant_text(text):
        st.session_state.current_assistant_text = text
        st.session_state.conversation.append({"role": "assistant", "text": text})
    
    def on_llm_start(user_text):
        # Show we're processing
        pass
    
    def on_metrics(metrics):
        st.session_state.metrics_history.append(metrics)
    
    def on_error(error):
        st.session_state.status = f"Error: {error}"
    
    return {
        "status": on_status,
        "user_text": on_user_text,
        "assistant_text": on_assistant_text,
        "llm_start": on_llm_start,
        "metrics": on_metrics,
        "error": on_error,
    }


def initialize_voice_handler(persona):
    if st.session_state.voice_handler:
        st.session_state.voice_handler.cleanup()
    
    callbacks = create_callbacks()
    st.session_state.voice_handler = SimpleVoiceHandler(persona, callbacks)
    st.session_state.selected_persona = persona
    st.session_state.status = "Ready"


def render_header():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="wide")
    st.title("AI Voice Assistant")

    st.markdown("""
        <style>
        .status-box {
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            text-align: center;
            margin: 20px 0;
        }
        .status-ready {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .status-recording {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            animation: pulse 1.5s infinite;
        }
        .status-processing {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .bubble-user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            font-size: 15px;
        }
        .bubble-assistant {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 8px 0;
            font-size: 15px;
        }
        .big-button {
            font-size: 18px !important;
            font-weight: 700 !important;
            padding: 15px 30px !important;
            border-radius: 12px !important;
            width: 100%;
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        </style>
    """, unsafe_allow_html=True)


def render_status_indicator():
    status = st.session_state.status
    
    if "Recording" in status:
        css_class = "status-recording"
    elif "Transcribing" in status or "Thinking" in status or "Speaking" in status:
        css_class = "status-processing"
    else:
        css_class = "status-ready"
    
    st.markdown(
        f'<div class="status-box {css_class}">{status}</div>',
        unsafe_allow_html=True
    )


def render_control_panel(personas):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        persona_names = [name for name, _ in personas]
        persona_map = {name: data for name, data in personas}
        
        selected_name = st.selectbox(
            "Support Scenario",
            persona_names,
            index=0
        )

        selected_persona = persona_map[selected_name]
        if (st.session_state.voice_handler is None or 
            st.session_state.selected_persona != selected_persona):
            initialize_voice_handler(selected_persona)

        scenario = selected_persona.get("scenario", "Support")
        st.info(f"**Active Scenario:** {scenario}")
    
    with col2:
        st.markdown("### Voice Controls")

        if not st.session_state.is_recording and not st.session_state.is_processing:
            if st.button("START RECORDING", key="record_btn", type="primary"):
                start_recording()
        
        # Stop button
        if st.session_state.is_recording:
            if st.button("STOP & PROCESS", key="stop_btn", type="secondary"):
                stop_and_process()
        
        # Reset button
        if st.button("New Conversation", key="reset_btn"):
            reset_conversation()


def start_recording():
    """Start recording audio"""
    if st.session_state.voice_handler:
        st.session_state.is_recording = True
        st.session_state.voice_handler.start_recording()
        st.session_state.status = "Recording - Speak now..."
        st.rerun()


def stop_and_process():
    """Stop recording and process the audio"""
    if st.session_state.voice_handler and st.session_state.is_recording:
        st.session_state.is_recording = False
        st.session_state.is_processing = True
        st.session_state.status = "Processing..."
        
        # Process synchronously (Streamlit doesn't handle threads well with session_state)
        with st.spinner("Processing your voice input..."):
            try:
                metrics = st.session_state.voice_handler.process_voice_input()
                if metrics and "error" not in metrics:
                    st.session_state.status = "Ready"
                else:
                    st.session_state.status = f"Error: {metrics.get('error', 'Unknown error')}"
            except Exception as e:
                st.session_state.status = f"Error: {str(e)}"
            finally:
                st.session_state.is_processing = False
        
        st.rerun()


def reset_conversation():
    """Reset the conversation"""
    if st.session_state.voice_handler:
        st.session_state.voice_handler.reset_conversation()
    st.session_state.conversation = []
    st.session_state.current_user_text = ""
    st.session_state.current_assistant_text = ""
    st.session_state.metrics_history = []
    st.session_state.status = "Ready"
    st.rerun()


def render_conversation():
    """Render conversation history"""
    st.markdown("### Conversation")
    
    if st.session_state.conversation:
        for turn in st.session_state.conversation:
            if turn["role"] == "user":
                st.markdown(
                    f'<div class="bubble-user">👤 You: {turn["text"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bubble-assistant">🤖 Assistant: {turn["text"]}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Click 'START RECORDING' to begin. Speak your question, then click 'STOP & PROCESS'.")


def render_metrics():
    """Render performance metrics"""
    if st.session_state.metrics_history:
        with st.expander("Performance Metrics", expanded=False):
            # Show latest metrics
            latest = st.session_state.metrics_history[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ASR", f"{latest.get('asr_ms', 0):.0f}ms")
            with col2:
                st.metric("LLM", f"{latest.get('llm_ms', 0):.0f}ms")
            with col3:
                st.metric("TTS", f"{latest.get('tts_ms', 0):.0f}ms")
            with col4:
                st.metric("Total", f"{latest.get('total_ms', 0):.0f}ms")
            
            # Show all metrics
            if len(st.session_state.metrics_history) > 1:
                st.markdown("**All Turns**")
                st.dataframe(
                    st.session_state.metrics_history,
                    hide_index=True
                )


def render_instructions():
    """Render usage instructions"""
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        ### Quick Guide
        
        1. **Select Scenario** - Choose the type of support needed
        2. **Click START RECORDING** - The microphone starts capturing
        3. **Speak Clearly** - Ask your question naturally
        4. **Click STOP & PROCESS** - Your audio will be:
           - Transcribed to text (ASR)
           - Processed by AI (LLM)
           - Converted to speech (TTS)
        5. **Listen to Response** - The AI will speak its answer
        6. **Continue** - Click START RECORDING again for follow-up questions
        
        ### Tips
        - Speak clearly and at normal pace
        - Wait for the "Recording" status before speaking
        - Click STOP when you finish your question
        - Use "New Conversation" to start fresh
        """)


def main():
    load_dotenv()
    init_session_state()
    render_header()

    personas = load_personas()
    if not personas:
        st.error("No personas found! Check the personas/ directory.")
        return

    render_status_indicator()

    render_control_panel(personas)
    
    st.markdown("---")

    render_conversation()

    render_metrics()

    render_instructions()
    
    if st.session_state.is_recording:
        time.sleep(0.3)
        st.rerun()

    st.markdown("---")

if __name__ == "__main__":
    main()
