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
from langsmith import traceable
from PIL import Image
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv      
# Load environment variables
load_dotenv()

#  API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY =  os.getenv("LANGSMITH_API_KEY")      


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
    st.info("Loading models from Hugging Face...")
    
    # Directly specify your Hugging Face model repository
    model_repo = "UtsavS/financial-earnings-call-classifier-final"
    
    try:
        # Load the classification model from Hugging Face
        classifier_model = AutoModelForSequenceClassification.from_pretrained(model_repo)
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
        # Load GPT-2 for text generation (reasoning)
        gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        return classifier_model, tokenizer, gpt2_model, gpt2_tokenizer

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.warning("Enabling test mode with mock results for UI demonstration.")
        st.session_state.test_mode = True
        return None, None, None, None

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

llm = ChatGroq(
    model="llama-3.1-8b-instant" ,#"deepseek-r1-distill-qwen-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
@traceable  # Add this decorator
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

@traceable  # Add this decorator
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



@traceable  # Add this decorator
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
                    st.stop()

        # Handle ongoing processing
        if 'processing_text' in st.session_state:
            state = st.session_state.processing_text
            current_sentence = state['sentences'][state['current_index']]
            
            try:
                with st.spinner(f"Analyzing sentence {state['current_index'] + 1} of {len(state['sentences'])}..."):
                    result = classify_and_reason(current_sentence, classifier_model, tokenizer)
                    state['results'].append(result)
                    state['current_index'] += 1
            except Exception as e:
                st.error(f"Error analyzing sentence: {str(e)}")
                del st.session_state.processing_text
                st.stop()

            # Check if all sentences have been processed
            if state['current_index'] >= len(state['sentences']):
                st.session_state.text_results = state['results']
                del st.session_state.processing_text
            else:
                st.rerun()
        # Handle analysis initiation
        # if analyze_button and text_input:
        #     with st.spinner("Preparing analysis..."):
        #         sentences = split_text_into_sentences(text_input)
        #         if sentences:
        #             st.session_state.processing_text = {
        #                 'sentences': sentences,
        #                 'current_index': 0,
        #                 'results': []
        #             }
        #         else:
        #             st.warning("No valid sentences found to analyze.")

        # # Handle ongoing processing
        # if 'processing_text' in st.session_state:
        #     state = st.session_state.processing_text
        #     current_sentence = state['sentences'][state['current_index']]
            
        #     with st.spinner(f"Analyzing sentence {state['current_index']+1} of {len(state['sentences'])}..."):
        #         result = classify_and_reason(current_sentence, classifier_model, tokenizer)
        #         state['results'].append(result)
        #         state['current_index'] += 1
            
        #     if state['current_index'] >= len(state['sentences']):
        #         st.session_state.text_results = state['results']
        #         del st.session_state.processing_text
        #     else:
        #         st.experimental_rerun()

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
            
            # # Show summary if all results are ready
            # if 'text_results' in st.session_state:
            #     st.subheader("Summary")
            #     fig1, chart, confidence_hist, prediction_counts = create_summary_charts(results)
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         st.pyplot(fig1)
            #     with col2:
            #         st.altair_chart(confidence_hist, use_container_width=True)
            #     st.dataframe(prediction_counts, use_container_width=True)

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
                st.rerun()

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
        # Custom CSS for styling
        st.markdown("""
        <style>
            .custom-expander {
                background-color: #f0f8ff; /* Light blue */
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .custom-expander:nth-child(2) {
                background-color: #fff0f5; /* Lavender blush */
            }
            .custom-expander:nth-child(3) {
                background-color: #f5f5dc; /* Beige */
            }
            .custom-expander:nth-child(4) {
                background-color: #e6e6fa; /* Lavender */
            }
            .custom-expander:nth-child(5) {
                background-color: #ffe4e1; /* Misty rose */
            }
            .custom-expander h3 {
                color: #333;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Demo Results Section
        with st.expander("About this tool 📊", expanded=False):
            st.markdown("""
            <div class="custom-expander">
                <h3>About this Tool</h3>
                <p>This tool is desgined to help earning call analyzers with intuitive and interactive interface !  </p>
            </div>
            """, unsafe_allow_html=True)
            # st.markdown("""
            # <div class="custom-expander">
            #     <h3>Demo Results</h3>
            #     <p>Below are some sample results from our tool:</p>
            #     <table style="width:100%; border-collapse:collapse;">
            #         <tr>
            #             <th style="border:1px solid #ddd; padding:8px;">Statement</th>
            #             <th style="border:1px solid #ddd; padding:8px;">Classification</th>
            #             <th style="border:1px solid #ddd; padding:8px;">Confidence Score</th>
            #         </tr>
            #         <tr>
            #             <td style="border:1px solid #ddd; padding:8px;">"We exceeded our revenue targets this quarter."</td>
            #             <td style="border:1px solid #ddd; padding:8px;">Answer</td>
            #             <td style="border:1px solid #ddd; padding:8px;">92%</td>
            #         </tr>
            #         <tr>
            #             <td style="border:1px solid #ddd; padding:8px;">"Thank you for joining the call."</td>
            #             <td style="border:1px solid #ddd; padding:8px;">Non-answer</td>
            #             <td style="border:1px solid #ddd; padding:8px;">85%</td>
            #         </tr>
            #     </table>
            # </div>
            # """, unsafe_allow_html=True)

        # About Team Section
        with st.expander("About Team 🌟", expanded=False):
            st.markdown("""
            <div class="custom-expander">
                <h3>About Our Team</h3>
                <p>Team XYZ is a group of passionate data scientists, financial analysts, and software engineers dedicated to building innovative tools for financial analysis.</p>
            </div>
            """, unsafe_allow_html=True)

        # Images Section
        with st.expander("Images 📸", expanded=False):
            st.markdown("""
            <div class="custom-expander">
                <h3>Gallery</h3>
                <p>Here are some images related to our project:</p>
            </div>
            """, unsafe_allow_html=True)
            # Load local images
            image1 = Image.open("images/team.jpg")  # Replace with your image path
            image2 = Image.open("images/team.jpg")  # Replace with your image path
            image3 = Image.open("images/team.jpg")  # Replace with your image path
            
            # Display images in a grid layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image1, caption="Image 1", use_column_width=True)
            with col2:
                st.image(image2, caption="Image 2", use_column_width=True)
            with col3:
                st.image(image3, caption="Image 3", use_column_width=True)

        # Links Section
        with st.expander("Useful Links 🔗", expanded=False):
            st.markdown("""
            <div class="custom-expander">
                <h3>Useful Links</h3>
                <ul>
                    <li>Presentation: <a href="https://github.com" target="_blank">Presentation</a></li>
                    <li>Fine-Tuning Code: <a href="https://github.com" target="_blank">Fine-Tuning Code</a></li>
                    <li>Web Intefce Code: <a href="https://github.com" target="_blank">Web Intefce Code</a></li>
                    <li>Video Submission: <a href="https://github.com" target="_blank">Video Submission</a></li>

            </div>
            """, unsafe_allow_html=True)

        # Contact Section
        with st.expander("Contact 📧", expanded=False):
            st.markdown("""
            <div class="custom-expander">
                <h3>Contact Us</h3>
                <p>If you have any questions or feedback, feel free to reach out:</p>
                <ul>
                    <li>Email: <a href="mailto:teamxyz@example.com">teamxyz@example.com</a></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


        # Disclaimer
        st.info("""
        **Note**: This is an analysis tool and should be used as one input among many when making financial decisions. 
        Always consult with qualified financial professionals before making investment decisions.
        """)

 
if __name__ == "__main__":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Earnings-Call"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    main()
