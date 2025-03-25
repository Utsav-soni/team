import os
import re
import io
import base64
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import altair as alt

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv      
# Load environment variables
load_dotenv()

#  API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY =  os.getenv("LANGSMITH_API_KEY")                 
LANGSMITH_TRACING = "true"

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY not set in the environment variables")
    st.error("API key for Groq is missing. Check your .env file.")
    st.stop()
    
    
# Set up page configuration
st.set_page_config(
    page_title="Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #F1F5F9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #3B82F6;
    }
    .answer-tag {
        background-color: #10B981;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .non-answer-tag {
        background-color: #EF4444;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define labels
LABELS = ["answer", "non-answer"]

@st.cache_resource
def load_models():
    """Load and cache the models to avoid reloading on each rerun"""
    st.info("Loading models (this may take a minute)...")
    
    # Define model path based on deployment environment
    # For local dev: use a local path
    # For deployment: use a path that works in the Streamlit environment
    model_path = os.getenv("MODEL_PATH", "financial-earnings-call-classifier-final")
    
    # Check if model exists, otherwise download from Hugging Face
    if not os.path.exists(model_path):
        st.warning(f"Model not found at {model_path}, using default model from Hugging Face")
        # Fall back to a default model - this is just an example, replace with your actual model on Hugging Face
        model_path = "distilbert-base-uncased"
    
    try:
        # Load the classification model
        classifier_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Try multiple approaches to load the tokenizer
        try:
            # Approach 1: Load directly from model path
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            try:
                # Approach 2: Try GPT2Tokenizer explicitly
                tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            except Exception:
                # Approach 3: Fall back to standard GPT2 tokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Load GPT-2 for text generation (reasoning)
        gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        return classifier_model, tokenizer, gpt2_model, gpt2_tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # Return dummy models for UI testing when real models can't be loaded
        if "test_mode" in st.session_state and st.session_state.test_mode:
            return None, None, None, None
        else:
            raise

def classify_text(text, classifier_model, tokenizer):
    """Classify a piece of text using the loaded model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    
    # Get predicted class
    prediction_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    predicted_class = np.argmax(prediction_probs)
    confidence = prediction_probs[0][predicted_class]
    
    return LABELS[predicted_class], confidence

# def generate_reasoning(text, prediction, gpt2_model, gpt2_tokenizer):
#     # """Generate an explanation for why the text was classified in a certain way"""
#     # reasoning_prompt = f"Explain in detail that why the statement: '{text}' was classified as '{prediction}' because "
#     # input_ids = gpt2_tokenizer.encode(reasoning_prompt, return_tensors="pt")
    
#     # with torch.no_grad():
#     #     output_ids = gpt2_model.generate(
#     #         input_ids, 
#     #         max_length=128, 
#     #         num_return_sequences=1, 
#     #         pad_token_id=gpt2_tokenizer.eos_token_id
#     #     )
    
#     # reasoning_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
#     # # Clean up the reasoning text a bit
#     # reasoning_text = reasoning_text.replace(reasoning_prompt, "")
    
#     return reasoning_text


llm = ChatGroq(
    model="llama-3.1-8b-instant" ,#"deepseek-r1-distill-qwen-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def generate_reasoning(text, prediction):
    messages = [
    {
        "role": "system", 
        "content": """
                    You are earning call analyzer and you are supporting the given statement
                    with given label, it could be answer or non-answer.
                    And your justification must be a reason that why that
                    statement is got particulat label by the custom fine-tuned model.
                   """
    },
    {
        "role": "user", 
        "content": f"""Explain why the given statement {text} is allocated under the label of {prediction}
                    from the earnings call analyzer finetuned model.
                    """
    }
    ]
    ai_msg = llm.invoke(messages)

    # Print response
    # print("AI Response:", ai_msg)
    return ai_msg.content

def classify_and_reason(text, classifier_model, tokenizer):
    """Classify text and generate reasoning in one step"""
    prediction, confidence = classify_text(text, classifier_model, tokenizer)
    
    # For testing when models aren't available
    if classifier_model is None:
        import random
        prediction = random.choice(LABELS)
        confidence = random.uniform(0.7, 0.99)
        reasoning = "This is a placeholder reasoning since models couldn't be loaded."
    else:
        # reasoning = generate_reasoning(text, prediction, gpt2_model, gpt2_tokenizer)
        reasoning = generate_reasoning(text, prediction)
    
    return {
        "text": text,
        "prediction": prediction,
        "confidence": float(confidence),
        "reasoning": reasoning
    }

def split_text_into_sentences(text):
    """Split text into sentences with improved handling of special cases"""
    # Cleanup text
    text = text.replace('\n', ' ').strip()
    
    # Handle common abbreviations to avoid splitting them
    text = re.sub(r'(Mr\.|Mrs\.|Dr\.|etc\.|i\.e\.|e\.g\.)', lambda m: m.group().replace('.', '<DOT>'), text)
    
    # Split by common sentence terminators
    raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    sentences = []
    for raw_sentence in raw_sentences:
        # Restore abbreviations
        raw_sentence = raw_sentence.replace('<DOT>', '.')
        
        # Skip empty sentences
        if raw_sentence.strip() and len(raw_sentence.strip()) > 10:
            sentences.append(raw_sentence.strip())
    
    return sentences

def process_text(text, classifier_model, tokenizer):
    """Process a block of text by splitting it into sentences and analyzing each one"""
    sentences = split_text_into_sentences(text)
    
    results = []
    for sentence in sentences:
        if len(sentence.strip()) > 10:  # Skip very short sentences
            result = classify_and_reason(sentence, classifier_model, tokenizer)
            results.append(result)
            
    return results

def get_confidence_class(confidence):
    """Return a CSS class based on confidence level"""
    if confidence >= 0.85:
        return "confidence-high"
    elif confidence >= 0.7:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_summary_charts(results):
    """Create summary charts from the analysis results"""
    # Create dataframe from results
    df = pd.DataFrame(results)
    
    # Count predictions
    prediction_counts = df['prediction'].value_counts().reset_index()
    prediction_counts.columns = ['Category', 'Count']
    
    # Calculate percentages
    total = prediction_counts['Count'].sum()
    prediction_counts['Percentage'] = (prediction_counts['Count'] / total * 100).round(1)
    
    # Create pie chart
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    colors = ['#10B981', '#EF4444'] if 'answer' in prediction_counts['Category'].values else ['#EF4444']
    ax1.pie(
        prediction_counts['Count'], 
        labels=prediction_counts['Category'], 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'width': 0.6}
    )
    ax1.axis('equal')
    plt.title('Distribution of Answers vs Non-Answers', size=16)
    
    # Create confidence distribution chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('prediction:N', title='Classification'),
        y=alt.Y('count():Q', title='Count'),
        color=alt.Color('prediction:N', 
                      scale=alt.Scale(domain=['answer', 'non-answer'], 
                                    range=['#10B981', '#EF4444'])),
        tooltip=['prediction:N', 'count():Q']
    ).properties(
        title='Classification Distribution'
    )
    
    # Create confidence histogram
    confidence_hist = alt.Chart(df).mark_bar().encode(
        x=alt.X('confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Level'),
        y=alt.Y('count():Q', title='Frequency'),
        color=alt.Color('prediction:N',
                      scale=alt.Scale(domain=['answer', 'non-answer'], 
                                    range=['#10B981', '#EF4444'])),
        tooltip=['confidence:Q', 'count():Q']
    ).properties(
        title='Confidence Distribution'
    )
    
    return fig1, chart, confidence_hist, prediction_counts

def create_results_table(results):
    """Create a formatted table of results"""
    if not results:
        return pd.DataFrame()
        
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Format confidence as percentage
    df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    
    # Rename columns for better display
    df = df.rename(columns={
        'text': 'Statement',
        'prediction': 'Classification',
        'confidence': 'Confidence',
        'reasoning': 'Reasoning'
    })
    
    return df

def create_download_link(df, filename="results.csv"):
    """Create a download link for the results"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download results as CSV</a>'
    return href

# def main():
#     """Main function to run the Streamlit app"""
#     # Test mode for when models can't be loaded
#     if "test_mode" not in st.session_state:
#         st.session_state.test_mode = False
    
#     # App header
#     st.markdown('<h1 class="main-header">Analyzer</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Analyze Distinguish between answers and non-answers</p>', unsafe_allow_html=True)
    
#     # Load models - handle errors gracefully
#     try:
#         classifier_model, tokenizer, gpt2_model, gpt2_tokenizer = load_models()
#     except Exception as e:
#         st.error(f"Failed to load models: {e}")
#         st.warning("Enabling test mode with mock results for UI demonstration.")
#         st.session_state.test_mode = True
#         classifier_model, tokenizer, gpt2_model, gpt2_tokenizer = None, None, None, None
    
#     # Create tabs
#     tab1, tab2, tab3 = st.tabs(["Analyze Text", "Analyze File", "About"])
    
#     # Tab 1: Analyze Text
#     with tab1:
#         st.subheader("📝 Analyze Specific Statements")
        
#         # Text input area
#         text_input = st.text_area(
#             "Enter a statement or paragraph to analyze:",
#             height=150,
#             placeholder="Paste the earnings call statement or paragraph here..."
#         )
        
#         col1, col2 = st.columns([1, 5])
#         with col1:
#             analyze_button = st.button("Analyze Text", type="primary")
        
#         # Process when button is clicked
#         if analyze_button and text_input:
#             with st.spinner("Analyzing text..."):
#                 results = process_text(text_input, classifier_model,tokenizer)
                
#                 # Store results in session state
#                 st.session_state.text_results = results
        
#         # Display results if available
#         if 'text_results' in st.session_state and st.session_state.text_results:
#             st.subheader("Analysis Results")
            
#             # Display each result in a card
#             for result in st.session_state.text_results:
#                 with st.container():
#                     st.markdown(f"""
#                     <div class="result-card">
#                         <p><strong>Statement:</strong> {result['text']}</p>
#                         <p>
#                             <strong>Classification:</strong> 
#                             <span class="{'answer-tag' if result['prediction'] == 'answer' else 'non-answer-tag'}">
#                                 {result['prediction'].upper()}
#                             </span>
#                             &nbsp;&nbsp;
#                             <strong>Confidence:</strong> 
#                             <span class="{get_confidence_class(result['confidence'])}">
#                                 {result['confidence']*100:.1f}%
#                             </span>
#                         </p>
#                         <p><strong>Reasoning:</strong> {result['reasoning']}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # Create summary if there are multiple results
#             if len(st.session_state.text_results) > 1:
#                 st.subheader("Summary")
                
#                 # Create visualization columns
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     fig1, chart, confidence_hist, prediction_counts = create_summary_charts(st.session_state.text_results)
#                     st.pyplot(fig1)
                
#                 with col2:
#                     st.altair_chart(confidence_hist, use_container_width=True)
                
#                 # Show stats as a table
#                 st.dataframe(prediction_counts, use_container_width=True)
    
#     # Tab 2: Analyze File
#     with tab2:
#         st.subheader("📄 Analyze Text File")
        
#         # File uploader
#         uploaded_file = st.file_uploader("Upload a text file:", type=['txt'])
        
#         if uploaded_file is not None:
#             # Read file content
#             file_content = uploaded_file.getvalue().decode("utf-8")
            
#             # Show a preview
#             with st.expander("File Preview", expanded=False):
#                 st.text(file_content)
            
#             # Analyze button
#             analyze_file_button = st.button("Analyze File", type="primary")
            
#             if analyze_file_button:
#                 with st.spinner("Analyzing file..."):
#                     results = process_text(file_content, classifier_model, tokenizer)
                    
#                     # Store results in session state
#                     st.session_state.file_results = results
        
#         # Display results if available
#         if 'file_results' in st.session_state and st.session_state.file_results:
#             st.subheader("Analysis Results")
            
#             # Create summary visualizations
#             fig1, chart, confidence_hist, prediction_counts = create_summary_charts(st.session_state.file_results)
            
#             # Use columns for charts
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.pyplot(fig1)
            
#             with col2:
#                 st.altair_chart(confidence_hist, use_container_width=True)
            
#             # Display summary stats
#             st.subheader("Summary Statistics")
#             st.dataframe(prediction_counts, use_container_width=True)
            
#             # Show detailed results in an expander
#             with st.expander("View All Results"):
#                 # Create a pandas DataFrame and display it
#                 results_df = create_results_table(st.session_state.file_results)
#                 st.dataframe(results_df, use_container_width=True)
                
#                 # Add download link for CSV
#                 st.markdown(create_download_link(results_df), unsafe_allow_html=True)
    
#     # Tab 3: About
#     with tab3:
#         st.subheader("ℹ️ About This Tool")
        
#         st.markdown("""                   
#         ### What This Tool Does
        
#         The Financial Earnings Call Analyzer helps you understand and analyze statements made during financial earnings calls. It uses natural language processing and machine learning to classify statements as:
        
#         - **Answers**: Substantive responses that provide useful information
#         - **Non-answers**: Evasive, vague, or non-informative responses
        
#         ### How It Works
        
#         This tool uses a fine-tuned transformer model trained on a dataset of earnings call transcripts. The model has learned patterns typical of substantive answers versus non-answers or evasive responses.
        
#         For each analyzed statement, the tool provides:
#         1. A classification (answer or non-answer)
#         2. A confidence score
#         3. A reasoning explanation for the classification
        
#         ### Use Cases
        
#         - Financial analysts reviewing earnings calls
#         - Investors conducting company research
#         - Corporate communications training
#         - Financial journalism
#         """)
        
#         st.info("""
#         **Note**: This is an analysis tool and should be used as one input among many when making financial decisions. 
#         Always consult with qualified financial professionals before making investment decisions.
#         """)




def main():
    """Main function to run the Streamlit app"""
    # Test mode for when models can't be loaded
    if "test_mode" not in st.session_state:
        st.session_state.test_mode = False

    # App header
    st.markdown('<h1 class="main-header">Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze Distinguish between answers and non-answers</p>', unsafe_allow_html=True)

    # Load models - handle errors gracefully
    try:
        classifier_model, tokenizer, gpt2_model, gpt2_tokenizer = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.warning("Enabling test mode with mock results for UI demonstration.")
        st.session_state.test_mode = True
        classifier_model, tokenizer, gpt2_model, gpt2_tokenizer = None, None, None, None

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Analyze Text", "Analyze File", "About"])

    # Tab 1: Analyze Text
    with tab1:
        st.subheader("📝 Analyze Specific Statements")
        text_input = st.text_area(
            "Enter a statement or paragraph to analyze:",
            height=150,
            placeholder="Paste the earnings call statement or paragraph here..."
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("Analyze Text", 
                                      type="primary", 
                                      disabled='processing_text' in st.session_state)

        # Handle analysis initiation
        if analyze_button and text_input:
            with st.spinner("Preparing analysis..."):
                sentences = split_text_into_sentences(text_input)
                if sentences:
                    st.session_state.processing_text = {
                        'sentences': sentences,
                        'current_index': 0,
                        'results': []
                    }
                else:
                    st.warning("No valid sentences found to analyze.")

        # Handle ongoing processing
        if 'processing_text' in st.session_state:
            state = st.session_state.processing_text
            current_sentence = state['sentences'][state['current_index']]
            
            with st.spinner(f"Analyzing sentence {state['current_index']+1} of {len(state['sentences'])}..."):
                result = classify_and_reason(current_sentence, classifier_model, tokenizer)
                state['results'].append(result)
                state['current_index'] += 1
            
            if state['current_index'] >= len(state['sentences']):
                st.session_state.text_results = state['results']
                del st.session_state.processing_text
            else:
                st.experimental_rerun()

        # Display results
        results = []
        if 'processing_text' in st.session_state:
            results = st.session_state.processing_text['results']
        elif 'text_results' in st.session_state:
            results = st.session_state.text_results

        if results:
            st.subheader("Analysis Results")
            for result in results:
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <p><strong>Statement:</strong> {result['text']}</p>
                        <p>
                            <strong>Classification:</strong> 
                            <span class="{'answer-tag' if result['prediction'] == 'answer' else 'non-answer-tag'}">
                                {result['prediction'].upper()}
                            </span>
                            &nbsp;&nbsp;
                            <strong>Confidence:</strong> 
                            <span class="{get_confidence_class(result['confidence'])}">
                                {result['confidence']*100:.1f}%
                            </span>
                        </p>
                        <p><strong>Reasoning:</strong> {result['reasoning']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show summary if all results are ready
            if 'text_results' in st.session_state:
                st.subheader("Summary")
                fig1, chart, confidence_hist, prediction_counts = create_summary_charts(results)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.altair_chart(confidence_hist, use_container_width=True)
                st.dataframe(prediction_counts, use_container_width=True)

    # # Tab 2: Analyze File
    # # with tab2:
    #     st.subheader("📁 Analyze Text File")
    #     uploaded_file = st.file_uploader("Upload a text file:", type=['txt'])
        
    #     if uploaded_file is not None:
    #         file_content = uploaded_file.getvalue().decode("utf-8")
            
    #         with st.expander("File Preview", expanded=False):
    #             st.text(file_content)
            
    #         analyze_file_button = st.button(
    #             "Analyze File", 
    #             type="primary",
    #             disabled='processing_file' in st.session_state
    #         )
            
    #         if analyze_file_button:
    #             with st.spinner("Preparing file analysis..."):
    #                 sentences = split_text_into_sentences(file_content)
    #                 if sentences:
    #                     st.session_state.processing_file = {
    #                         'sentences': sentences,
    #                         'current_index': 0,
    #                         'results': []
    #                     }
    #                 else:
    #                     st.warning("No valid sentences found in file.")

    #     # Handle file processing
    #     if 'processing_file' in st.session_state:
    #         state = st.session_state.processing_file
    #         current_sentence = state['sentences'][state['current_index']]
            
    #         with st.spinner(f"Analyzing sentence {state['current_index']+1} of {len(state['sentences'])}..."):
    #             result = classify_and_reason(current_sentence, classifier_model, tokenizer)
    #             state['results'].append(result)
    #             state['current_index'] += 1
            
    #         if state['current_index'] >= len(state['sentences']):
    #             st.session_state.file_results = state['results']
    #             del st.session_state.processing_file
    #         else:
    #             st.experimental_rerun()

    #     # Display file results
    #     file_results = []
    #     if 'processing_file' in st.session_state:
    #         file_results = st.session_state.processing_file['results']
    #     elif 'file_results' in st.session_state:
    #         file_results = st.session_state.file_results

    #     if file_results:
    #         st.subheader("Analysis Results")
    #         for result in file_results:
    #             with st.container():
    #                 st.markdown(f"""
    #                 <div class="result-card">
    #                     <p><strong>Statement:</strong> {result['text']}</p>
    #                     <p>
    #                         <strong>Classification:</strong> 
    #                         <span class="{'answer-tag' if result['prediction'] == 'answer' else 'non-answer-tag'}">
    #                             {result['prediction'].upper()}
    #                         </span>
    #                         &nbsp;&nbsp;
    #                         <strong>Confidence:</strong> 
    #                         <span class="{get_confidence_class(result['confidence'])}">
    #                             {result['confidence']*100:.1f}%
    #                         </span>
    #                     </p>
    #                     <p><strong>Reasoning:</strong> {result['reasoning']}</p>
    #                 </div>
    #                 """, unsafe_allow_html=True)
            
    #         # Show summary if all results are ready
    #         if 'file_results' in st.session_state:
    #             st.subheader("Summary")
    #             fig1, chart, confidence_hist, prediction_counts = create_summary_charts(file_results)
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 st.pyplot(fig1)
    #             with col2:
    #                 st.altair_chart(confidence_hist, use_container_width=True)
    #             st.dataframe(prediction_counts, use_container_width=True)
    #             st.markdown(create_download_link(create_results_table(file_results)), unsafe_allow_html=True)
    with tab2:
        st.subheader("📁 Analyze Text File")
        uploaded_file = st.file_uploader("Upload a text file:", type=['txt'])
        
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            
            with st.expander("File Preview", expanded=False):
                st.text(file_content)
            
            analyze_file_button = st.button(
                "Analyze File", 
                type="primary",
                disabled='processing_file' in st.session_state
            )
            
            if analyze_file_button:
                with st.spinner("Preparing file analysis..."):
                    sentences = split_text_into_sentences(file_content)
                    if sentences:
                        st.session_state.processing_file = {
                            'sentences': sentences,
                            'current_index': 0,
                            'results': []
                        }
                    else:
                        st.warning("No valid sentences found in file.")

        # Handle file processing
        if 'processing_file' in st.session_state:
            state = st.session_state.processing_file
            current_sentence = state['sentences'][state['current_index']]
            
            with st.spinner(f"Analyzing sentence {state['current_index']+1} of {len(state['sentences'])}..."):
                result = classify_and_reason(current_sentence, classifier_model, tokenizer)
                state['results'].append(result)
                state['current_index'] += 1
            
            if state['current_index'] >= len(state['sentences']):
                st.session_state.file_results = state['results']
                del st.session_state.processing_file
            else:
                st.experimental_rerun()

        # Display file results
        file_results = []
        if 'processing_file' in st.session_state:
            file_results = st.session_state.processing_file['results']
        elif 'file_results' in st.session_state:
            file_results = st.session_state.file_results
#here
        if file_results:
            st.subheader("Analysis Results")
            with st.expander("View All Results"):
                results_df = create_results_table(file_results)
                st.dataframe(results_df, use_container_width=True)
                st.markdown(create_download_link(results_df), unsafe_allow_html=True)
            
            # Preserve original summary interface
            if 'file_results' in st.session_state:
                st.subheader("Summary")
                fig1, chart, confidence_hist, prediction_counts = create_summary_charts(file_results)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.altair_chart(confidence_hist, use_container_width=True)
                st.dataframe(prediction_counts, use_container_width=True)
            
            for result in file_results:
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <p><strong>Statement:</strong> {result['text']}</p>
                        <p>
                            <strong>Classification:</strong> 
                            <span class="{'answer-tag' if result['prediction'] == 'answer' else 'non-answer-tag'}">
                                {result['prediction'].upper()}
                            </span>
                            &nbsp;&nbsp;
                            <strong>Confidence:</strong> 
                            <span class="{get_confidence_class(result['confidence'])}">
                                {result['confidence']*100:.1f}%
                            </span>
                        </p>
                        <p><strong>Reasoning:</strong> {result['reasoning']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
          
                

    # Tab 3: About
    with tab3:
        st.subheader("ℹ️ About This Tool")
        st.markdown("""
        ### This tool is developed by Team: xyz.
        ### What This Tool Does
        The Financial Earnings Call Analyzer helps you understand and analyze statements made during financial earnings calls. It uses gpt2 fintuned on binary classification dataset to classify statements as:
        - **Answers**: Substantive responses that provide useful information
        - **Non-answers**: Evasive, vague, greetings, neutral or non-informative responses
        ### How It Works
        This tool uses a fine-tuned transformer model gpt2 trained on a dataset of earnings call transcripts. The model has learned patterns typical of substantive answers versus non-answers or evasive responses.
        For each analyzed statement, the tool provides:
        1. A classification (answer or non-answer)
        2. A confidence score
        3. A reasoning explanation for the classification
        ### Use Cases
        - Financial analysts reviewing earnings calls
        - Investors conducting company research
        - Corporate communications training
        - Financial journalism
        """)
        st.info("""
        **Note**: This is an analysis tool and should be used as one input among many when making financial decisions. 
        Always consult with qualified financial professionals before making investment decisions.
        """)
        
if __name__ == "__main__":
    main()
