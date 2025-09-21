import streamlit as st
import os

# Disable Streamlit's file watcher to avoid PyTorch compatibility issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

from sample_outputs import generate_text

# Configure the Streamlit page
st.set_page_config(
    page_title="Transformer Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for professional interface
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 12px 16px;
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #404040;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
        outline: none;
    }
    
    /* AI response box */
    .output-box {
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        margin: 16px 0;
        border: 1px solid #333333;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* User input box */
    .user-input-box {
        background-color: #2a2a2a;
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        margin: 16px 0;
        border: 1px solid #404040;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Send button styling */
    .stButton > button {
        background-color: #4a90e2;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.2s ease;
        font-size: 16px;
    }
    
    .stButton > button:hover {
        background-color: #357abd;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    
    /* Clear button styling */
    .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: #999999;
        border: 1px solid #404040;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #2a2a2a;
        color: #ffffff;
        border-color: #666666;
    }
    
    /* Title styling */
    .main-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #ffffff !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #4a90e2;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    # Parameters
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=2000, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    top_k = st.slider("Top-K", min_value=1, max_value=100, value=40, step=1)
    top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    
    st.markdown("---")
    
    # Model info
    st.subheader("🤖 Model Info")
    st.write("**Vocab Size**: 8,192")
    st.write("**Embedding Dim**: 512")
    st.write("**Sequence Length**: 1,024")
    st.write("**Heads**: 8")
    st.write("**Layers**: 21")

# Main title
st.markdown('<h1 class="main-title">🤖 Transformer Chat</h1>', unsafe_allow_html=True)

# Display area for current conversation
if "current_input" not in st.session_state:
    st.session_state.current_input = ""
if "current_output" not in st.session_state:
    st.session_state.current_output = ""

# Show current conversation if exists
if st.session_state.current_input:
    st.markdown(f"""
    <div class="user-input-box">
        <strong>👤 You</strong><br><br>
        {st.session_state.current_input}
    </div>
    """, unsafe_allow_html=True)

if st.session_state.current_output:
    st.markdown(f"""
    <div class="output-box">
        <strong>🤖 AI Assistant</strong><br><br>
        {st.session_state.current_output}
    </div>
    """, unsafe_allow_html=True)

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Input area at bottom - fixed position
st.markdown("---")
st.subheader("💬 Type your message:")

# Input area
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Enter your message here...",
        key="chat_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send 🚀", type="primary", use_container_width=True)

# Handle sending message
if send_button and user_input.strip():
    # Store user input
    st.session_state.current_input = user_input
    
    # Generate AI response
    with st.spinner("🤖 Generating response..."):
        try:
            generated_text = generate_text(
                user_input,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Store AI response
            st.session_state.current_output = generated_text
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.current_output = f"Sorry, I encountered an error: {str(e)}"
    
    # Rerun to show the conversation
    st.rerun()

# Clear conversation button
if st.button("🗑️ Clear Conversation", type="secondary"):
    st.session_state.current_input = ""
    st.session_state.current_output = ""
    st.rerun()