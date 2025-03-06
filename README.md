# **Playground AI**

## Description:
Playground AI is an innovative ensemble AI model project that aggregates responses from multiple AI models available on the OpenRouter API. It provides users with a unified interface where they can ask questions, and the system will fetch answers from various AI models, analyze them, and deliver the most optimal response.

The system works as follows:
1. **Question Input**: Users input their questions.
2. **Model Query**: The system simultaneously queries multiple AI models (like Mistral, Zephyr, Llama-2, etc.) through the OpenRouter API.
3. **Response Analysis**: The system analyzes the responses using clustering techniques to find the most consistent and optimal answer.
4. **Output**: The system returns the best answer along with metadata about the contributing models.

### Features:
- Query multiple AI models simultaneously
- Analyze and cluster responses
- Deliver the most optimal answer
- Track model contributions and response rates
- Modern UI interface

### Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/playground-ai.git
   cd playground-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenRouter API Key:
   - Create a `secrets.toml` file in your project directory.
   - Add your API key:
     ```toml
     OPENROUTER_API_KEY = "your-api-key-here"
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Usage:
1. Open the application in your browser.
2. Enter your question in the input field.
3. The system will fetch responses from multiple AI models and display the most optimal answer.

### Requirements:
- Python 3.8+
- Streamlit
- aiohttp
- numpy
- scikit-learn
- sentence-transformers
- transformers

### Contributing:
Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request.

### License:
[MIT License](LICENSE)

### Contact:
For any questions or feedback, feel free to contact us at your.email@example.com.

---

### Special Thanks:
- OpenRouter API for providing access to multiple AI models.
- Hugging Face for the Sentence Transformers library.
```

---

### Step-by-Step Instructions to Resolve Errors:

#### **1. Install Required Packages**
Run the following command in your project directory to install all dependencies:

```bash
pip install streamlit aiohttp numpy scikit-learn sentence-transformers
```

#### **2. Create `secrets.toml` File**
Create a `secrets.toml` file in your project directory with your OpenRouter API key:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

#### **3. Fix Code Errors**
Modify your `app.py` file as follows:

```python
import streamlit as st
import asyncio
import aiohttp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Configuration
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
TIMEOUT_SECONDS = 10
MAX_MODELS = 5

MODELS = {
    "mistralai/mistral-7b-instruct": {"name": "Mistral 7B"},
    "huggingfaceh4/zephyr-7b-beta": {"name": "Zephyr 7B"},
    "meta-llama/llama-2-13b-chat": {"name": "Llama 2 13B"},
    "google/palm-2-chat-bison": {"name": "Palm 2 Chat"},
    "nousresearch/nous-hermes-llama2-13b": {"name": "Nous Hermes 13B"}
}

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

async def query_model(session, model_id, prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://your-site.com",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return model_id, data['choices'][0]['message']['content']
            return model_id, None
    except:
        return model_id, None

async def get_responses(prompt):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for model_id in MODELS:
            task = asyncio.wait_for(
                query_model(session, model_id, prompt),
                timeout=TIMEOUT_SECONDS
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

def analyze_responses(responses):
    valid_responses = [r for r in responses if not isinstance(r, Exception) and r[1]]
    
    if not valid_responses:
        return {
            "final_answer": "No valid responses received",
            "supporting_models": [],
            "total_responses": 0,
            "total_models_tried": len(MODELS)
        }
    
    texts = [r[1] for r in valid_responses]
    models = [MODELS[r[0]]['name'] for r in valid_responses]
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(texts)
    
    pca = PCA(n_components=2)
    reduced_embeds = pca.fit_transform(embeddings)
    
    n_clusters = min(3, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_embeds)
    
    cluster_counts = dict(zip(*np.unique(clusters, return_counts=True)))
    main_cluster = max(cluster_counts, key=cluster_counts.get)
    
    cluster_responses = [texts[i] for i, c in enumerate(clusters) if c == main_cluster]
    cluster_models = [models[i] for i, c in enumerate(clusters) if c == main_cluster]
    
    return {
        "final_answer": cluster_responses[0],
        "supporting_models": cluster_models,
        "total_responses": len(valid_responses),
        "total_models_tried": len(MODELS)
    }

# UI Components
st.set_page_config(page_title="Playground AI", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput input {
        padding: 1rem !important;
        border-radius: 15px !important;
    }
    .css-1cpxqw2 {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #f0f2f6 !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .bot-message {
        background-color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSpinner > div {
        margin: 2rem auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Playground AI")
st.caption("Advanced answers powered by multiple AI models collaborating")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(f"Consensus from: {', '.join(message['meta']['models'])}")

# Input area
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.spinner(f"Consulting {len(MODELS)} AI models..."):
        responses = asyncio.run(get_responses(prompt))
    
    with st.spinner("Analyzing consensus..."):
        result = analyze_responses(responses)
    
    # Add assistant response
    with st.chat_message("assistant"):
        st.markdown(result["final_answer"])
        st.caption(f"Consensus from: {', '.join(result['supporting_models'])}")
        with st.expander("Technical details"):
            st.write(f"Response rate: {result['total_responses']}/{result['total_models_tried']}")
            st.write("Models contributing to consensus:")
            st.json(result["supporting_models"])
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_answer"],
        "meta": {
            "models": result["supporting_models"],
            "response_rate": f"{result['total_responses']}/{result['total_models_tried']}"
        }
    })
```

#### **4. Additional Fixes**
- **Error loading embedding model**: Ensure that the `sentence-transformers` library is installed and the model is downloaded. Run:
  ```bash
  pip install sentence-transformers
  ```
- **KeyError: 'total_responses'**: The code has been updated to always include `total_responses` and `total_models_tried` in the result dictionary.

#### **5. Running the Application**
Run the following command in your project directory:
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to use the application.

---

### GitHub Repository Setup:
1. Create a new repository named `playground-ai`.
2. Upload the following files:
   - `app.py`: Your main application file.
   - `secrets.toml`: **Note**: Do not include your API key in this file when uploading to GitHub. Keep it local.
   - `requirements.txt`: Contains all the necessary package dependencies.
   - `README.md`: Documentation file.
   - `LICENSE`: MIT License file.

3. Ignore unnecessary files (e.g., `.cache`, virtual environments) by creating a `.gitignore` file.

---

### Final Steps:
- Ensure your API key is stored securely (preferably in a `.env` file or through environment variables).
- Test the application thoroughly before deployment.
- Follow GitHub best practices for sharing open-source projects.
