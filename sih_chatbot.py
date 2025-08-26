import PyPDF2
import pandas as pd
import os
from dotenv import load_dotenv
try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

# Load environment variables
load_dotenv()

# Extract text from PDF
def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# Read Excel file
def read_excel_data(excel_path):
    """Read Excel file and return DataFrame"""
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return pd.DataFrame()

# Enhanced chatbot logic with OpenAI integration
def answer_question(question, pdf_text, excel_df, use_openai=False):
    """Answer questions using keyword matching or OpenAI"""
    question = question.lower()
    
    # Try OpenAI if available and requested
    if use_openai and openai_available and os.getenv("OPENAI_API_KEY"):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            # Prepare context for OpenAI
            context = f"""
            PDF Content (first 2000 chars): {pdf_text[:2000]}
            Excel Data Summary: {excel_df.head(10).to_string() if not excel_df.empty else 'No data'}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Smart India Hackathon problem statements. Answer questions based on the provided PDF and Excel data. Be specific and helpful."},
                    {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
                ],
                max_tokens=500
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenAI error: {e}. Using fallback method.")
    
    # Fallback: Enhanced keyword search
    if any(word in question for word in ["problem statement", "problem", "statements"]):
        if not excel_df.empty and 'Problem Statement' in excel_df.columns:
            statements = excel_df['Problem Statement'].dropna().head(10).tolist()
            return f"Available problem statements:\n" + "\n".join([f"{i+1}. {stmt}" for i, stmt in enumerate(statements)])
        else:
            # Search in PDF for problem statements
            lines = [line.strip() for line in pdf_text.split('\n') if 'problem' in line.lower() and len(line.strip()) > 10]
            return "Problem statements found in PDF:\n" + "\n".join(lines[:5]) if lines else "No problem statements found."
    
    elif any(word in question for word in ["winner", "winners", "winning"]):
        if not excel_df.empty:
            if 'Winners' in excel_df.columns:
                total_winners = excel_df['Winners'].sum()
                return f"Total winners across all categories: {total_winners}"
            elif 'Winner' in excel_df.columns:
                total_winners = excel_df['Winner'].sum()
                return f"Total winners across all categories: {total_winners}"
        return "Winner information not found in the data."
    
    elif any(word in question for word in ["enrollment", "enrollments", "participants"]):
        if not excel_df.empty:
            if 'Enrollments' in excel_df.columns:
                total_enrollments = excel_df['Enrollments'].sum()
                return f"Total enrollments: {total_enrollments}"
            elif 'Participants' in excel_df.columns:
                total_participants = excel_df['Participants'].sum()
                return f"Total participants: {total_participants}"
        return "Enrollment information not found in the data."
    
    elif any(word in question for word in ["field", "category", "domain"]):
        if not excel_df.empty:
            fields = excel_df.columns.tolist()
            return f"Available fields/categories in the data: {', '.join(fields)}"
        return "Field information not available."
    
    elif any(word in question for word in ["choose", "select", "recommend"]):
        recommendations = []
        if not excel_df.empty:
            # Try to find columns that might indicate popularity or success
            for col in ['Enrollments', 'Participants', 'Success Rate', 'Difficulty']:
                if col in excel_df.columns:
                    top_items = excel_df.nlargest(3, col)
                    recommendations.append(f"Top 3 by {col}:")
                    for idx, row in top_items.iterrows():
                        recommendations.append(f"- {row.get('Problem Statement', f'Row {idx}')}")
        
        if recommendations:
            return "\n".join(recommendations)
        else:
            return "I recommend looking at problem statements that align with your skills and interests. Check the enrollment numbers and difficulty levels to make an informed choice."
    
    else:
        # General search in both PDF and Excel
        search_terms = question.split()
        relevant_lines = []
        
        # Search PDF
        for line in pdf_text.split('\n'):
            if any(term in line.lower() for term in search_terms) and len(line.strip()) > 10:
                relevant_lines.append(line.strip())
        
        # Search Excel
        if not excel_df.empty:
            for col in excel_df.columns:
                if excel_df[col].dtype == 'object':  # Text columns
                    matches = excel_df[excel_df[col].astype(str).str.contains('|'.join(search_terms), case=False, na=False)]
                    if not matches.empty:
                        relevant_lines.extend(matches[col].astype(str).tolist())
        
        if relevant_lines:
            return "Found relevant information:\n" + "\n".join(relevant_lines[:5])
        else:
            return "Sorry, I couldn't find specific information about that. Try asking about problem statements, winners, enrollments, or fields."

def main():
    """Main function to run the command-line chatbot"""
    pdf_path = 'test.pdf'
    excel_path = 'SIH_PS_2024.xlsx'
    
    print("🤖 SIH Problem Statement Chatbot")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found. Please place it in the same folder as this script.")
        return
    
    if not os.path.exists(excel_path):
        print(f"❌ Error: {excel_path} not found. Please place it in the same folder as this script.")
        return
    
    # Load data
    print("📄 Loading PDF and Excel files...")
    pdf_text = extract_pdf_text(pdf_path)
    excel_df = read_excel_data(excel_path)
    
    if not pdf_text.strip():
        print("⚠️  Warning: PDF appears to be empty or unreadable.")
    
    if excel_df.empty:
        print("⚠️  Warning: Excel file appears to be empty or unreadable.")
    
    print("✅ Files loaded successfully!")
    print("\n💡 You can ask questions like:")
    print("- Which problem statement should I choose?")
    print("- How many winners are there?")
    print("- What are the enrollment numbers?")
    print("- Show me problem statements in AI/ML domain")
    
    # Check for OpenAI
    use_openai = False
    if openai_available and os.getenv("OPENAI_API_KEY"):
        use_openai = input("\n🤖 OpenAI API key detected. Use AI for smarter answers? (y/n): ").lower().startswith('y')
    elif openai_available:
        print("\n💡 Tip: Set OPENAI_API_KEY environment variable for AI-powered answers!")
    
    print(f"\n🚀 Chatbot ready! {'(AI-powered)' if use_openai else '(Keyword-based)'}")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            
            print("🔍 Searching for answer...")
            answer = answer_question(question, pdf_text, excel_df, use_openai)
            print(f"\n💬 Answer:\n{answer}\n")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try asking your question differently.\n")

if __name__ == "__main__":
    main()
