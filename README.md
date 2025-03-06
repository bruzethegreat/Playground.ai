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
For any questions or feedback, feel free to contact us at bruze999@gmail.com

---

### Special Thanks:
- OpenRouter API for providing access to multiple AI models.
- Hugging Face for the Sentence Transformers library.

### Step-by-Step Setup Instructions:

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

#### **3. Running the Application**
Run the following command in your project directory:
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to use the application.

---

