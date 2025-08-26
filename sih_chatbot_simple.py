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
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def read_excel_data(excel_path):
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return pd.DataFrame()

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
            return f"OpenAI error: {e}. Using fallback search."
    
    # Fallback: keyword search
    q = question.lower()
    if "problem statement" in q or "problem" in q:
        if not excel_df.empty:
            # Look for problem statement columns
            for col in excel_df.columns:
                if 'problem' in col.lower() or 'statement' in col.lower():
                    statements = excel_df[col].dropna().head(5).tolist()
                    return f"Available problem statements:\n" + "\n".join([f"{i+1}. {stmt}" for i, stmt in enumerate(statements)])
        return "Problem statements found in your data. Please check your Excel file for detailed information."
    
    elif "winner" in q:
        if not excel_df.empty:
            winner_cols = [col for col in excel_df.columns if 'winner' in col.lower()]
            if winner_cols:
                total_winners = excel_df[winner_cols[0]].sum()
                return f"Total winners: {total_winners}"
        return "Winner information not found in the current data."
    
    elif "enrollment" in q or "participant" in q:
        if not excel_df.empty:
            enroll_cols = [col for col in excel_df.columns if any(word in col.lower() for word in ['enrollment', 'participant', 'registration'])]
            if enroll_cols:
                total_enrollments = excel_df[enroll_cols[0]].sum()
                return f"Total enrollments/participants: {total_enrollments}"
        return "Enrollment information not found in the current data."
    
    elif "domain" in q or "field" in q or "category" in q:
        if not excel_df.empty:
            return f"Available fields/categories in your data: {', '.join(excel_df.columns.tolist())}"
        return "Domain information not available."
    
    elif "recommend" in q or "choose" in q or "suggest" in q:
        return "Based on your data, I recommend looking at problem statements that match your skills and have good enrollment numbers. Check the Excel file for specific details about each problem statement."
    
    else:
        # General search
        search_terms = q.split()
        results = []
        
        # Search PDF
        if pdf_text:
            for line in pdf_text.split('\n'):
                if any(term in line.lower() for term in search_terms) and len(line.strip()) > 10:
                    results.append(line.strip())
                    if len(results) >= 3:
                        break
        
        # Search Excel
        if not excel_df.empty:
            for col in excel_df.columns:
                if excel_df[col].dtype == 'object':
                    matches = excel_df[excel_df[col].astype(str).str.contains('|'.join(search_terms), case=False, na=False)]
                    if not matches.empty:
                        results.extend(matches[col].astype(str).head(2).tolist())
        
        if results:
            return "Found relevant information:\n" + "\n".join(results[:5])
        else:
            return "Sorry, I couldn't find specific information about that. Try asking about problem statements, winners, enrollments, or domains."

def main():
    st.set_page_config(page_title="SIH Chatbot", page_icon="🤖", layout="wide")
    
    st.title("🤖 SIH Problem Statement Chatbot")
    st.write("Ask questions about SIH problem statements, winners, enrollments, and more!")
    
    # File paths
    pdf_path = 'test.pdf'
    excel_path = 'SIH_PS_2024.xlsx'
    
    # Check files
    pdf_exists = os.path.exists(pdf_path)
    excel_exists = os.path.exists(excel_path)
    
    col1, col2 = st.columns(2)
    with col1:
        if pdf_exists:
            st.success("✅ test.pdf found")
        else:
            st.error("❌ test.pdf not found")
    
    with col2:
        if excel_exists:
            st.success("✅ SIH_PS_2024.xlsx found")
        else:
            st.error("❌ SIH_PS_2024.xlsx not found")
    
    # Load data
    pdf_text = ""
    excel_df = pd.DataFrame()
    
    if pdf_exists:
        with st.spinner("Loading PDF..."):
            pdf_text = extract_pdf_text(pdf_path)
    
    if excel_exists:
        with st.spinner("Loading Excel..."):
            excel_df = read_excel_data(excel_path)
            if not excel_df.empty:
                st.success(f"📊 Excel loaded: {len(excel_df)} rows, {len(excel_df.columns)} columns")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        if openai_available:
            st.subheader("AI Settings")
            use_openai = st.checkbox("Use OpenAI for smart answers")
            if use_openai:
                openai_api_key = st.text_input("OpenAI API Key", type="password")
            else:
                openai_api_key = None
        else:
            st.warning("Install 'openai' package for AI-powered answers")
            use_openai = False
            openai_api_key = None
    
    # Main interface
    st.markdown("---")
    st.subheader("💬 Ask Your Question")
    
    # Question input
    question = st.text_input(
        "Enter your question here:",
        placeholder="e.g., Which problem statement should I choose?",
        help="Ask about problem statements, winners, enrollments, recommendations, etc."
    )
    
    # Example questions
    st.write("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Which problem to choose?"):
            question = "Which problem statement should I choose?"
    
    with col2:
        if st.button("🏆 How many winners?"):
            question = "How many winners are there?"
    
    with col3:
        if st.button("📊 Show enrollments"):
            question = "What are the enrollment numbers?"
    
    # Process question
    if question:
        with st.spinner("Searching for answer..."):
            if pdf_exists or excel_exists:
                answer = smart_answer(question, pdf_text, excel_df, use_openai, openai_api_key)
                st.markdown("### 📝 Answer:")
                st.write(answer)
            else:
                st.error("Please make sure both test.pdf and SIH_PS_2024.xlsx are in the same folder as this app.")
    
    # Data preview
    if not excel_df.empty:
        st.markdown("---")
        with st.expander("📊 Preview Excel Data"):
            st.dataframe(excel_df.head(10))

if __name__ == "__main__":
    main()
