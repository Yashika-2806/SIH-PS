def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text
def read_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    return df
def smart_answer(question, pdf_text, excel_df):
    # If you want to use OpenAI, uncomment below and set your API key
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "system", "content": "You are a helpful assistant for SIH problem statements."},
    #               {"role": "user", "content": f"PDF: {pdf_text}\nEXCEL: {excel_df.to_string()}\nQuestion: {question}"}]
    # )
    # return response['choices'][0]['message']['content']
    # Fallback: keyword search
    q = question.lower()
    if "problem statement" in q:
        return "Here are some problem statements: " + ", ".join(excel_df['Problem Statement'].head(5).astype(str))
    elif "winner" in q:
        winners = excel_df['Winners'].sum() if 'Winners' in excel_df.columns else 'Data not available'
        return f"Total winners: {winners}"
    elif "enrollment" in q:
        enrollments = excel_df['Enrollments'].sum() if 'Enrollments' in excel_df.columns else 'Data not available'
        return f"Total enrollments: {enrollments}"
    else:
        lines = [line for line in pdf_text.split('\n') if any(word in line.lower() for word in q.split())]
        return '\n'.join(lines) if lines else "Sorry, I couldn't find an answer."
def main():
    st.title("SIH Problem Statement Chatbot")
    st.write("Ask questions about SIH problem statements, winners, enrollments, and more!")
    pdf_path = 'test.pdf'
    excel_path = 'SIH_PS_2024.xlsx'
    if not os.path.exists(pdf_path) or not os.path.exists(excel_path):
        st.error("Please make sure 'test.pdf' and 'SIH_PS_2024.xlsx' are in the same folder as this app.")
        return
    pdf_text = extract_pdf_text(pdf_path)
    excel_df = read_excel_data(excel_path)
    question = st.text_input("Your question:")
    if question:
        answer = smart_answer(question, pdf_text, excel_df)
        st.write("**Answer:**", answer)

import streamlit as st
import PyPDF2
import pandas as pd
import os
try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def read_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    return df

def smart_answer(question, pdf_text, excel_df, use_openai=False, openai_api_key=None):
    if use_openai and openai_available and openai_api_key:
        try:
            openai.api_key = openai_api_key
            context = f"PDF Content:\n{pdf_text[:2000]}...\nEXCEL Data:\n{excel_df.head(10).to_string()}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for SIH problem statements. Answer based on the provided PDF and Excel data."},
                    {"role": "user", "content": f"{context}\nQuestion: {question}"}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"OpenAI error: {e}. Falling back to keyword search."
    # Fallback: keyword search
    q = question.lower()
    if "problem statement" in q:
        return "Here are some problem statements: " + ", ".join(excel_df['Problem Statement'].head(5).astype(str))
    elif "winner" in q:
        winners = excel_df['Winners'].sum() if 'Winners' in excel_df.columns else 'Data not available'
        return f"Total winners: {winners}"
    elif "enrollment" in q:
        enrollments = excel_df['Enrollments'].sum() if 'Enrollments' in excel_df.columns else 'Data not available'
        return f"Total enrollments: {enrollments}"
    else:
        lines = [line for line in pdf_text.split('\n') if any(word in line.lower() for word in q.split())]
        return '\n'.join(lines) if lines else "Sorry, I couldn't find an answer."

def main():
    st.title("SIH Problem Statement Chatbot")
    st.write("Ask questions about SIH problem statements, winners, enrollments, and more!")
    st.info("Place 'test.pdf' and 'SIH_PS_2024.xlsx' in this folder. Optionally, set your OpenAI API key in the sidebar for smarter answers.")
    
    pdf_path = 'test.pdf'
    excel_path = 'SIH_PS_2024.xlsx'
    
    # Check file existence
    files_exist = os.path.exists(pdf_path) and os.path.exists(excel_path)
    
    if not files_exist:
        st.error("Please make sure 'test.pdf' and 'SIH_PS_2024.xlsx' are in the same folder as this app.")
        st.write("**Missing files:**")
        if not os.path.exists(pdf_path):
            st.write(f"❌ {pdf_path}")
        else:
            st.write(f"✅ {pdf_path}")
        if not os.path.exists(excel_path):
            st.write(f"❌ {excel_path}")
        else:
            st.write(f"✅ {excel_path}")
    else:
        st.success("✅ Files found! Loading data...")
    
    # Load data if files exist
    pdf_text = ""
    excel_df = pd.DataFrame()
    
    if files_exist:
        try:
            pdf_text = extract_pdf_text(pdf_path)
            excel_df = read_excel_data(excel_path)
            st.success("📄 Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading files: {e}")
    
    # Sidebar for OpenAI settings
    use_openai = False
    openai_api_key = None
    if openai_available:
        with st.sidebar:
            st.header("AI Settings")
            use_openai = st.checkbox("Use OpenAI for smart answers", value=False)
            if use_openai:
                openai_api_key = st.text_input("OpenAI API Key", type="password")
    else:
        st.sidebar.warning("Install openai package for AI-powered answers.")
    
    # Always show the input field
    st.markdown("---")
    
    # Check if a question was clicked
    default_question = ""
    if 'clicked_question' in st.session_state:
        default_question = st.session_state['clicked_question']
        del st.session_state['clicked_question']
    
    question = st.text_input("Your question:", value=default_question, placeholder="e.g., Which problem statement should I choose?")
    
    if question:
        if files_exist and (pdf_text or not excel_df.empty):
            answer = smart_answer(question, pdf_text, excel_df, use_openai, openai_api_key)
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.warning("Please ensure both PDF and Excel files are available to get answers.")
    
    # Show example questions
    st.markdown("---")
    st.markdown("**💡 Example Questions:**")
    example_questions = [
        "Which problem statement should I choose?",
        "How many winners are there?",
        "What are the enrollment numbers?",
        "Show me available domains",
        "Recommend a problem for beginners"
    ]
    
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        with cols[i]:
            if st.button(eq, key=f"example_{i}"):
                # Set the question in session state instead of query params
                st.session_state['clicked_question'] = eq
                st.rerun()

if __name__ == "__main__":
    main()
