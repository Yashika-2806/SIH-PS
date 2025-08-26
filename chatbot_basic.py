import streamlit as st
import PyPDF2
import pandas as pd
import os

# Simple function to read PDF
def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except:
        text = "Error reading PDF"
    return text

# Simple function to read Excel
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except:
        return pd.DataFrame()

# Simple answer function
def get_answer(question, pdf_text, excel_df):
    q = question.lower()
    
    if "problem statement" in q or "problem" in q:
        return "Here are the problem statements available in your data. Check your Excel file for detailed problem statements."
    
    elif "winner" in q:
        if not excel_df.empty:
            return f"Total data rows in Excel: {len(excel_df)}"
        return "Winner data not available"
    
    elif "enrollment" in q:
        if not excel_df.empty:
            return f"Excel contains {len(excel_df)} rows of data with {len(excel_df.columns)} columns"
        return "Enrollment data not available"
    
    elif "recommend" in q or "choose" in q:
        return "Based on your interests and skills, choose a problem statement that aligns with your expertise. Check the Excel file for detailed information."
    
    else:
        return f"I found some information about '{question}'. Please check your PDF and Excel files for detailed data."

# Main app
st.title("🤖 SIH Problem Statement Chatbot")
st.write("Ask questions about your SIH data!")

# Check files
pdf_exists = os.path.exists('test.pdf')
excel_exists = os.path.exists('SIH_PS_2024.xlsx')

if pdf_exists:
    st.success("✅ test.pdf found")
else:
    st.error("❌ test.pdf not found")

if excel_exists:
    st.success("✅ SIH_PS_2024.xlsx found")
else:
    st.error("❌ SIH_PS_2024.xlsx not found")

# Load data
pdf_text = ""
excel_df = pd.DataFrame()

if pdf_exists:
    pdf_text = read_pdf('test.pdf')
    st.info("📄 PDF loaded")

if excel_exists:
    excel_df = read_excel('SIH_PS_2024.xlsx')
    if not excel_df.empty:
        st.info(f"📊 Excel loaded: {len(excel_df)} rows")

# Question input
st.markdown("---")
question = st.text_input("Ask your question:", placeholder="e.g., Which problem statement should I choose?")

# Quick buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Problem Statements"):
        question = "show me problem statements"
with col2:
    if st.button("Winners Info"):
        question = "how many winners"
with col3:
    if st.button("Recommendations"):
        question = "recommend a problem"

# Show answer
if question:
    st.markdown("### Answer:")
    answer = get_answer(question, pdf_text, excel_df)
    st.write(answer)

# Show data preview
if not excel_df.empty:
    st.markdown("---")
    st.markdown("### 📊 Excel Data Preview:")
    st.dataframe(excel_df.head())
