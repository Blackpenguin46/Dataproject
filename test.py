import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import requests

st.set_page_config(page_title="📊 Combined Dashboard", layout="centered")

# --- Custom CSS for centering and settings icon ---
st.markdown("""
    <style>
    .centered-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }
    .settings-icon {
        position: absolute;
        top: 1.5rem;
        right: 2.5rem;
        font-size: 1.7rem;
        cursor: pointer;
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

# --- Settings Modal State ---
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

def show_settings_modal():
    with st.modal("Settings", key="settings-modal"):
        st.write("(Add your settings here)")
        if st.button("Close"):
            st.session_state.show_settings = False

# --- Settings Icon ---
st.markdown(
    f"<span class='settings-icon' onclick=\"window.dispatchEvent(new CustomEvent('openSettings'))\">⚙️</span>",
    unsafe_allow_html=True
)

# --- JS to trigger Streamlit rerun for settings icon ---
st.markdown(
    """
    <script>
    window.addEventListener('openSettings', function() {
        window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', key: 'show_settings', value: true}, '*');
    });
    </script>
    """,
    unsafe_allow_html=True
)

# --- Listen for settings icon click ---
if st.session_state.get("show_settings", False):
    show_settings_modal()

# --- Constants ---
DATA_FILE = "process_data.json"

# --- Helper Functions ---
@st.cache_data
def load_data():
    """Loads process data from a JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    """Saves process data to a JSON file."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- BPO Dashboard Functions ---
def display_bpo_dashboard():
    """Displays the Business Process Optimization Dashboard."""
    st.title("📊 Business Process Optimization Dashboard")

    with st.form("process_form", clear_on_submit=True):
        st.subheader("➕ Add New Process")
        name = st.text_input("Process Name")
        issues = st.text_area("Current Issues")
        goals = st.text_area("Optimization Goals")
        submitted = st.form_submit_button("Submit Process")

        if submitted:
            if name and issues and goals:
                data = load_data()
                data.append({"name": name, "issues": issues, "goals": goals})
                save_data(data)
                st.success(f"✅ Process \'{name}\' added successfully!")
                st.cache_data.clear()
            else:
                st.warning("Please complete all fields.")

    data = load_data()
    st.subheader("📋 Submitted Processes")
    if data:
        for i, entry in enumerate(data):
            with st.expander(f"[{i+1}] {entry['name']}"):
                st.write(f"**Issues:** {entry['issues']}")
                st.write(f"**Goals:** {entry['goals']}")
                if st.button(f"🗑️ Delete Process {i+1}", key=f"del{i}"):
                    data.pop(i)
                    save_data(data)
                    st.cache_data.clear()
                    st.experimental_rerun()
    else:
        st.info("No processes submitted yet.")

    st.subheader("📈 Most Common Optimization Goals")
    if data:
        df = pd.DataFrame(data)
        goal_counts = df['goals'].value_counts().reset_index()
        goal_counts.columns = ['Goal', 'Frequency']
        fig = px.bar(goal_counts, x='Goal', y='Frequency', title="Most Common Goals")
        st.plotly_chart(fig)

    st.subheader("📤 Export Data")
    if data:
        df = pd.DataFrame(data)
        st.download_button("Download as CSV", data=df.to_csv(index=False), file_name="bpo_data.csv", mime="text/csv")

# --- Data Science Assistant Functions ---
@st.cache_data
def train_model(df, features, target):
    """Trains a Linear Regression model and returns performance metrics."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, model

def display_data_science_assistant():
    """Displays the Data Science Assistant."""
    st.title("🧠 Data Science Assistant")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview", df.head())

        columns = df.columns.tolist()
        ID = st.selectbox("🆔 Select ID Column", columns)
        target = st.selectbox("🎯 Select Target Variable", columns)
        features = st.multiselect("🧮 Select Features", [col for col in columns if col not in [target, ID]])

        if features:
            r2, mse, model = train_model(df, features, target)

            st.subheader("📊 Model Performance")
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**MSE:** {mse:.2f}")

            st.subheader("📈 Feature vs Target")
            selected_feature = st.selectbox("Select feature to plot", features)
            fig = px.scatter(df, x=selected_feature, y=target, title=f"{selected_feature} vs {target}")
            st.plotly_chart(fig)
        else:
            st.info("Select at least one feature to train the model.")

# --- Chatbot Functions (Ollama) ---
def get_ollama_response(prompt, model="mistral"):
    """Gets a response from the Ollama local LLM."""
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": st.session_state.messages,
        "stream": False  # Ensure Ollama returns a single JSON object
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.ok:
            result = response.json()
            return result["message"]["content"]
        else:
            st.error(f"Ollama error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return None

def display_chatbot():
    """Displays the Ollama Chatbot."""
    st.title("🤖 Ask Me Anything (Ollama Chatbot)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("💬 Ask your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = get_ollama_response(prompt)
            if assistant_response:
                full_response += assistant_response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Main Application Logic ---
st.markdown("<div class='centered-nav'>", unsafe_allow_html=True)
page = st.radio(
    "",
    ["📈 BPO Dashboard", "🧠 Data Science Assistant", "🤖 Chatbot"],
    horizontal=True
)
st.markdown("</div>", unsafe_allow_html=True)

if page == "📈 BPO Dashboard":
    display_bpo_dashboard()
elif page == "🧠 Data Science Assistant":
    display_data_science_assistant()
elif page == "🤖 Chatbot":
    display_chatbot()