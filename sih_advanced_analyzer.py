import streamlit as st
import pandas as pd
import PyPDF2
import os
import re
import json
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from textblob import TextBlob
from fuzzywuzzy import fuzz, process
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SIHAnalyzer:
    def __init__(self):
        self.data = {}
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.problem_vectors = None
        
        # Machine Learning Success Predictor
        self.success_model = None
        self.feature_names = []
        
        # Advanced analytics cache
        self.analytics_cache = {}
        
        # Competition intelligence
        self.competition_db = {
            'AI/ML': {'avg_teams': 85, 'winning_threshold': 88},
            'Blockchain': {'avg_teams': 45, 'winning_threshold': 82},
            'IoT': {'avg_teams': 65, 'winning_threshold': 84},
            'Healthcare': {'avg_teams': 70, 'winning_threshold': 86},
            'Smart Cities': {'avg_teams': 55, 'winning_threshold': 83},
            'Cybersecurity': {'avg_teams': 60, 'winning_threshold': 87},
            'Default': {'avg_teams': 40, 'winning_threshold': 80}
        }
        
        # Success factors database
        self.success_factors = {
            'technical_innovation': 0.25,
            'market_relevance': 0.20,
            'implementation_feasibility': 0.20,
            'team_skill_match': 0.15,
            'presentation_quality': 0.10,
            'social_impact': 0.10
        }
        
        # Technology trend tracking
        self.tech_trends = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai', 'ml', 'nlp', 'computer vision'],
            'Blockchain': ['blockchain', 'cryptocurrency', 'smart contract', 'decentralized', 'web3', 'crypto'],
            'IoT': ['iot', 'sensor', 'monitoring', 'smart device', 'connected', 'automation', 'embedded'],
            'Cloud': ['cloud', 'aws', 'azure', 'microservices', 'serverless', 'containers', 'kubernetes'],
            'Cybersecurity': ['security', 'cyber', 'encryption', 'authentication', 'privacy', 'secure', 'vulnerability'],
            'Data Science': ['data science', 'analytics', 'big data', 'visualization', 'statistics', 'insights'],
            'Mobile': ['mobile', 'android', 'ios', 'app development', 'flutter', 'react native'],
            'Web': ['web', 'frontend', 'backend', 'api', 'javascript', 'react', 'angular', 'vue'],
            'AR/VR': ['augmented reality', 'virtual reality', 'ar', 'vr', 'mixed reality', 'metaverse'],
            'Robotics': ['robotics', 'automation', 'robot', 'autonomous', 'drone', 'mechanical']
        }
        
    def extract_excel_data(self, excel_file):
        """Extract data from uploaded Excel file"""
        try:
            df = pd.read_excel(excel_file)
            return df
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
            return pd.DataFrame()
    
    def train_success_predictor(self):
        """Train ML model to predict hackathon success"""
        if not self.data:
            return
            
        from sklearn.ensemble import RandomForestClassifier
        
        # Generate training data from historical problems
        features = []
        labels = []
        
        for year, problems in self.data.items():
            for problem in problems:
                # Extract features
                text = (problem['title'] + ' ' + problem['description']).lower()
                
                feature_vector = [
                    len(text.split()),  # Text length
                    self.calculate_difficulty_score(problem),
                    len(re.findall(r'\btechnolog\w+', text)),  # Tech mentions
                    1 if 'ai' in text else 0,
                    1 if 'blockchain' in text else 0,
                    1 if 'mobile' in text else 0,
                    1 if 'web' in text else 0,
                    len([word for word in text.split() if len(word) > 10]),  # Complex words
                ]
                
                features.append(feature_vector)
                # Simulate success label based on difficulty and relevance
                success_score = (100 - self.calculate_difficulty_score(problem) * 5 + 
                               len(text.split()) * 0.1) / 2
                labels.append(1 if success_score > 60 else 0)
        
        if len(features) > 10:  # Need minimum data
            self.success_model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.success_model.fit(features, labels)
            self.feature_names = ['text_length', 'difficulty', 'tech_mentions', 'has_ai', 
                                'has_blockchain', 'has_mobile', 'has_web', 'complex_words']
    
    def calculate_difficulty_score(self, problem):
        """Enhanced difficulty calculation"""
        text = (problem['title'] + ' ' + problem['description']).lower()
        
        # Technical complexity keywords with weights
        complexity_indicators = {
            'very_high': {'keywords': ['quantum', 'distributed systems', 'neural architecture', 'deep reinforcement'], 'weight': 5},
            'high': {'keywords': ['machine learning', 'blockchain', 'ai', 'real-time', 'scalable', 'microservices'], 'weight': 4},
            'medium_high': {'keywords': ['api integration', 'cloud native', 'computer vision', 'nlp'], 'weight': 3},
            'medium': {'keywords': ['web application', 'mobile app', 'database', 'dashboard'], 'weight': 2},
            'low': {'keywords': ['website', 'form', 'basic', 'simple', 'static'], 'weight': 1}
        }
        
        score = 0
        for level, data in complexity_indicators.items():
            for keyword in data['keywords']:
                if keyword in text:
                    score += data['weight']
        
        # Normalize to 1-10 scale
        return min(10, max(1, int(score * 0.8 + 2)))
    
    def get_ml_success_prediction(self, problem):
        """Get ML-based success prediction"""
        if not self.success_model:
            self.train_success_predictor()
            
        if not self.success_model:
            return 50  # Default
            
        text = (problem['title'] + ' ' + problem['description']).lower()
        feature_vector = [
            len(text.split()),
            self.calculate_difficulty_score(problem),
            len(re.findall(r'\btechnolog\w+', text)),
            1 if 'ai' in text else 0,
            1 if 'blockchain' in text else 0,
            1 if 'mobile' in text else 0,
            1 if 'web' in text else 0,
            len([word for word in text.split() if len(word) > 10]),
        ]
        
        try:
            probability = self.success_model.predict_proba([feature_vector])[0][1] * 100
            return min(95, max(5, probability))
        except:
            return 50
    
    def generate_hackathon_strategy(self, problem, team_profile):
        """Generate winning strategy for hackathon"""
        strategy = {
            'preparation_phase': [],
            'development_phase': [],
            'presentation_phase': [],
            'risk_mitigation': [],
            'competitive_advantages': []
        }
        
        difficulty = self.calculate_difficulty_score(problem)
        text = (problem['title'] + ' ' + problem['description']).lower()
        
        # Preparation strategies
        if difficulty >= 8:
            strategy['preparation_phase'].extend([
                "🎯 Conduct deep technical research (1-2 weeks before)",
                "💡 Build proof-of-concept early",
                "📚 Study similar existing solutions",
                "🤝 Identify potential mentors/experts"
            ])
        else:
            strategy['preparation_phase'].extend([
                "📋 Focus on implementation planning",
                "🎨 Prepare UI/UX mockups",
                "🔧 Set up development environment",
                "📊 Research target user needs"
            ])
        
        # Development strategies based on problem type
        if any(tech in text for tech in ['ai', 'ml', 'machine learning']):
            strategy['development_phase'].extend([
                "🧠 Start with simple baseline model",
                "📊 Focus on data quality over model complexity",
                "🎯 Prepare demo dataset",
                "📈 Show clear accuracy metrics"
            ])
        
        if 'web' in text or 'platform' in text:
            strategy['development_phase'].extend([
                "🚀 Prioritize working MVP over features",
                "💎 Focus on user experience",
                "🔄 Implement core user journey first",
                "📱 Ensure responsive design"
            ])
        
        # Presentation strategies
        strategy['presentation_phase'].extend([
            "🌟 Lead with problem statement and impact",
            "📊 Show quantifiable results/metrics",
            "🎭 Prepare compelling demo narrative",
            "💡 Highlight unique technical innovation",
            "📈 Present market potential and scalability"
        ])
        
        # Risk mitigation based on difficulty
        if difficulty >= 7:
            strategy['risk_mitigation'].extend([
                "⚠️ Have backup simpler solution ready",
                "🔧 Test critical components early",
                "📋 Prepare for technical Q&A",
                "🤝 Define clear team responsibilities"
            ])
        
        return strategy
    
    def parse_excel_problems(self, df, year):
        """Parse problem statements from Excel DataFrame"""
        problems = []
        
        # Common column mappings for SIH Excel files
        column_mappings = {
            'id': ['id', 'problem id', 'ps id', 'statement id', 'problem statement id', 'sl no', 'sr no', 'no', 'number'],
            'title': ['title', 'problem statement', 'problem title', 'statement', 'description', 'problem description'],
            'category': ['category', 'theme', 'domain', 'area', 'sector', 'field'],
            'ministry': ['ministry', 'organization', 'dept', 'department', 'nodal ministry'],
            'type': ['type', 'software/hardware', 'problem type', 'category type'],
            'complexity': ['complexity', 'difficulty', 'level']
        }
        
        # Find the best matching columns
        matched_columns = {}
        df_columns_lower = [col.lower().strip() for col in df.columns]
        
        for field, possible_names in column_mappings.items():
            best_match = None
            for possible_name in possible_names:
                for i, col in enumerate(df_columns_lower):
                    if possible_name in col or col in possible_name:
                        best_match = df.columns[i]
                        break
                if best_match:
                    break
            matched_columns[field] = best_match
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract basic information
                problem_id = str(row[matched_columns['id']]) if matched_columns['id'] else f"PS_{year}_{idx+1}"
                
                # Try to get title/description
                title = ""
                description = ""
                
                if matched_columns['title']:
                    title_text = str(row[matched_columns['title']]) if pd.notna(row[matched_columns['title']]) else ""
                    if len(title_text) > 100:
                        title = title_text[:100] + "..."
                        description = title_text
                    else:
                        title = title_text
                        description = title_text
                
                # Get category
                category = str(row[matched_columns['category']]) if matched_columns['category'] and pd.notna(row[matched_columns['category']]) else 'Other'
                
                # Get ministry/organization
                ministry = str(row[matched_columns['ministry']]) if matched_columns['ministry'] and pd.notna(row[matched_columns['ministry']]) else 'Unknown'
                
                # Get type (Software/Hardware)
                problem_type = str(row[matched_columns['type']]) if matched_columns['type'] and pd.notna(row[matched_columns['type']]) else 'Unknown'
                
                # Auto-categorize if category is generic
                if category.lower() in ['other', 'unknown', 'nan']:
                    category = self.categorize_problem(description)
                
                # Estimate complexity if not provided
                complexity = str(row[matched_columns['complexity']]) if matched_columns['complexity'] and pd.notna(row[matched_columns['complexity']]) else self.estimate_complexity(description)
                
                if title and len(title.strip()) > 5:  # Valid problem statement
                    problems.append({
                        'id': problem_id.strip(),
                        'title': title.strip(),
                        'description': description.strip(),
                        'category': category.strip(),
                        'ministry': ministry.strip(),
                        'type': problem_type.strip(),
                        'year': year,
                        'complexity': complexity,
                        'keywords': self.extract_keywords(description),
                        'source': 'excel'
                    })
                    
            except Exception as e:
                continue  # Skip problematic rows
        
        return problems
        """Extract text from uploaded PDF file"""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def parse_problem_statements(self, text, year):
        """Parse problem statements from PDF text"""
        problems = []
        
        # Enhanced regex patterns for different formats
        patterns = [
            r'(?:PS\s*ID|Problem\s*Statement\s*ID|ID)[:\s]*([A-Z0-9]+).*?(?:Problem\s*Statement|Title)[:\s]*(.*?)(?=(?:PS\s*ID|Problem\s*Statement\s*ID|ID|$))',
            r'(\d+)\.\s*(.*?)(?=\d+\.|$)',
            r'SIH(\d+)[:\s]*(.*?)(?=SIH\d+|$)',
            r'([A-Z]{3}\d+)[:\s]*(.*?)(?=[A-Z]{3}\d+|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    if len(match) == 2:
                        ps_id, description = match
                        if len(description.strip()) > 20:  # Filter out very short descriptions
                            problems.append({
                                'id': ps_id.strip(),
                                'title': description.strip()[:200],  # First 200 chars as title
                                'description': description.strip(),
                                'year': year,
                                'category': self.categorize_problem(description.strip()),
                                'complexity': self.estimate_complexity(description.strip()),
                                'keywords': self.extract_keywords(description.strip())
                            })
                break
        
        return problems
    
    def categorize_problem(self, text):
        """Automatically categorize problem statements"""
        categories = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai', 'ml'],
            'Healthcare': ['health', 'medical', 'patient', 'disease', 'medicine', 'healthcare', 'telemedicine'],
            'Agriculture': ['farm', 'crop', 'agriculture', 'farmer', 'soil', 'irrigation', 'pest'],
            'Education': ['education', 'student', 'learning', 'school', 'college', 'university', 'teaching'],
            'Smart Cities': ['smart city', 'urban', 'traffic', 'transportation', 'municipal', 'civic'],
            'Environment': ['environment', 'pollution', 'waste', 'clean', 'green', 'sustainable', 'renewable'],
            'Fintech': ['financial', 'banking', 'payment', 'blockchain', 'cryptocurrency', 'fintech'],
            'IoT': ['iot', 'sensor', 'monitoring', 'smart device', 'connected', 'automation'],
            'Blockchain': ['blockchain', 'distributed ledger', 'smart contract', 'decentralized'],
            'Cybersecurity': ['security', 'cyber', 'encryption', 'authentication', 'privacy', 'secure']
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'Other'
    
    def estimate_complexity(self, text):
        """Estimate problem complexity based on text analysis"""
        complexity_indicators = {
            'High': ['complex', 'advanced', 'sophisticated', 'machine learning', 'ai', 'blockchain', 'real-time', 'scalable'],
            'Medium': ['system', 'platform', 'application', 'integration', 'database', 'api', 'web'],
            'Low': ['simple', 'basic', 'monitor', 'track', 'display', 'report', 'dashboard']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for level, indicators in complexity_indicators.items():
            scores[level] = sum(1 for indicator in indicators if indicator in text_lower)
        
        if scores['High'] > 0:
            return 'High'
        elif scores['Medium'] > scores['Low']:
            return 'Medium'
        else:
            return 'Low'
    
    def extract_keywords(self, text):
        """Extract important keywords from text"""
        try:
            blob = TextBlob(text)
            # Get noun phrases and important words
            keywords = []
            for phrase in blob.noun_phrases:
                if len(phrase.split()) <= 3:  # Keep short phrases
                    keywords.append(phrase)
            
            # Add important single words
            words = [word.lower() for word in blob.words if len(word) > 4]
            keywords.extend(words[:10])  # Top 10 words
            
            return list(set(keywords))[:15]  # Return unique keywords, max 15
        except:
            return []
    
    def add_data(self, year, problems):
        """Add problem statements for a specific year"""
        self.data[year] = problems
        
    def get_similarity_matrix(self):
        """Calculate similarity between all problem statements"""
        all_problems = []
        for year, problems in self.data.items():
            all_problems.extend(problems)
        
        if not all_problems:
            return None
        
        descriptions = [p['description'] for p in all_problems]
        self.problem_vectors = self.vectorizer.fit_transform(descriptions)
        similarity_matrix = cosine_similarity(self.problem_vectors)
        
        return similarity_matrix, all_problems
    
    def find_similar_problems(self, problem_text, threshold=0.3):
        """Find similar problems across years"""
        if not self.data:
            return []
        
        all_problems = []
        for year, problems in self.data.items():
            all_problems.extend(problems)
        
        if not all_problems:
            return []
        
        descriptions = [p['description'] for p in all_problems]
        all_descriptions = descriptions + [problem_text]
        
        vectors = self.vectorizer.fit_transform(all_descriptions)
        similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()
        
        similar_problems = []
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                similar_problems.append({
                    'problem': all_problems[i],
                    'similarity': similarity
                })
        
        return sorted(similar_problems, key=lambda x: x['similarity'], reverse=True)
    
    def get_trending_categories(self):
        """Analyze trending categories across years"""
        category_trends = {}
        
        for year, problems in self.data.items():
            categories = [p['category'] for p in problems]
            category_counts = Counter(categories)
            category_trends[year] = category_counts
        
        return category_trends
    
    def recommend_problems(self, interests, skills, team_size=4, experience_level='Medium'):
        """Recommend problems based on user preferences"""
        recommendations = []
        
        for year, problems in self.data.items():
            for problem in problems:
                score = 0
                
                # Interest matching
                problem_text = (problem['title'] + ' ' + problem['description']).lower()
                for interest in interests:
                    if interest.lower() in problem_text:
                        score += 3
                
                # Skill matching
                for skill in skills:
                    if skill.lower() in problem_text:
                        score += 2
                
                # Experience level matching
                if problem['complexity'] == experience_level:
                    score += 2
                elif (experience_level == 'High' and problem['complexity'] == 'Medium') or \
                     (experience_level == 'Medium' and problem['complexity'] in ['Low', 'High']):
                    score += 1
                
                # Category bonus
                if any(cat.lower() in problem_text for cat in interests):
                    score += 1
                
                if score > 0:
                    recommendations.append({
                        'problem': problem,
                        'score': score,
                        'match_reason': self.get_match_reason(problem, interests, skills)
                    })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def get_match_reason(self, problem, interests, skills):
        """Get reason for recommendation"""
        reasons = []
        problem_text = (problem['title'] + ' ' + problem['description']).lower()
        
        for interest in interests:
            if interest.lower() in problem_text:
                reasons.append(f"Matches your interest in {interest}")
        
        for skill in skills:
            if skill.lower() in problem_text:
                reasons.append(f"Utilizes your {skill} skills")
        
        if problem['complexity']:
            reasons.append(f"Complexity level: {problem['complexity']}")
        
        return "; ".join(reasons) if reasons else "General match"

# Initialize the analyzer
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SIHAnalyzer()

def main():
    st.set_page_config(
        page_title="SIH Problem Statement Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Advanced SIH Problem Statement Analyzer")
    st.markdown("### Multi-Year Analysis & Intelligent Recommendation System")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🔧 Navigation")
        page = st.selectbox("Choose a feature:", [
            "📁 Data Upload",
            "📊 Data Overview", 
            "🔍 Problem Search",
            "📈 Trend Analysis",
            "🎯 Recommendations",
            "📋 Compare Problems",
            "🎖️ Strategy Generator",  # New amazing feature!
            "💡 Insights Dashboard"
        ])
    
    if page == "📁 Data Upload":
        show_upload_page()
    elif page == "📊 Data Overview":
        show_overview_page()
    elif page == "🔍 Problem Search":
        show_search_page()
    elif page == "📈 Trend Analysis":
        show_trends_page()
    elif page == "🎯 Recommendations":
        show_recommendations_page()
    elif page == "📋 Compare Problems":
        show_comparison_page()
    elif page == "🎖️ Strategy Generator":
        show_strategy_generator_page()
    elif page == "💡 Insights Dashboard":
        show_insights_page()

def show_upload_page():
    st.header("📁 Upload SIH Problem Statement Files")
    st.markdown("Upload PDF files from different SIH years to build your comprehensive database.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose Excel or PDF files", 
            type=['xlsx', 'xls', 'pdf'], 
            accept_multiple_files=True,
            help="Upload Excel or PDF files containing SIH problem statements from different years"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                year = st.text_input(f"Year for {uploaded_file.name}:", 
                                   value="2024", 
                                   key=f"year_{uploaded_file.name}")
                
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        problems = []
                        
                        if file_extension in ['xlsx', 'xls']:
                            # Process Excel file
                            df = st.session_state.analyzer.extract_excel_data(uploaded_file)
                            if not df.empty:
                                problems = st.session_state.analyzer.parse_excel_problems(df, year)
                                st.success(f"✅ Processed {len(problems)} problem statements from Excel file ({year})")
                                
                                # Show Excel preview
                                st.subheader(f"Excel Data Preview ({year}):")
                                st.dataframe(df.head())
                                
                        elif file_extension == 'pdf':
                            # Process PDF file
                            text = st.session_state.analyzer.extract_pdf_text(uploaded_file)
                            if text:
                                problems = st.session_state.analyzer.parse_problem_statements(text, year)
                                st.success(f"✅ Processed {len(problems)} problem statements from PDF ({year})")
                        
                        if problems:
                            st.session_state.analyzer.add_data(year, problems)
                            
                            # Show preview of processed problems
                            st.subheader(f"Processed Problems Preview ({year}):")
                            preview_df = pd.DataFrame(problems[:5])
                            display_columns = ['id', 'title', 'category', 'complexity', 'type'] if 'type' in preview_df.columns else ['id', 'title', 'category', 'complexity']
                            st.dataframe(preview_df[display_columns])
                        else:
                            st.warning("No valid problem statements found in this file.")
    
    with col2:
        st.info("💡 **Tips:**\n\n"
                "• Upload Excel files (.xlsx, .xls) or PDFs from multiple years\n"
                "• Excel files are automatically parsed for problem statements\n"
                "• Use clear year labels (e.g., 2024, 2023)\n"
                "• The system will automatically extract and categorize problems")
        
        if st.session_state.analyzer.data:
            st.success(f"📊 **Current Data:**\n\n"
                      f"Years loaded: {len(st.session_state.analyzer.data)}\n\n"
                      f"Total problems: {sum(len(problems) for problems in st.session_state.analyzer.data.values())}")

def show_overview_page():
    st.header("📊 Data Overview")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload PDF files first.")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_problems = sum(len(problems) for problems in st.session_state.analyzer.data.values())
    years_count = len(st.session_state.analyzer.data)
    
    # Count Excel vs PDF sources
    excel_count = 0
    pdf_count = 0
    for problems in st.session_state.analyzer.data.values():
        for problem in problems:
            if problem.get('source') == 'excel':
                excel_count += 1
            else:
                pdf_count += 1
    
    with col1:
        st.metric("Total Problems", total_problems)
    with col2:
        st.metric("Years Covered", years_count)
    with col3:
        categories = set()
        for problems in st.session_state.analyzer.data.values():
            categories.update(p['category'] for p in problems)
        st.metric("Categories", len(categories))
    with col4:
        st.metric("Excel Sources", excel_count, f"PDF: {pdf_count}")
    
    # Year-wise breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Year-wise Distribution")
        year_data = []
        for year, problems in st.session_state.analyzer.data.items():
            year_data.append({'Year': year, 'Count': len(problems)})
        
        if year_data:
            df = pd.DataFrame(year_data)
            fig = px.bar(df, x='Year', y='Count', title="Problems per Year")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏷️ Category Distribution")
        all_categories = []
        for problems in st.session_state.analyzer.data.values():
            all_categories.extend(p['category'] for p in problems)
        
        if all_categories:
            category_counts = Counter(all_categories)
            fig = px.pie(values=list(category_counts.values()), 
                        names=list(category_counts.keys()), 
                        title="Problem Categories")
            st.plotly_chart(fig, use_container_width=True)
    
    # Complexity analysis
    st.subheader("⚡ Complexity Analysis")
    complexity_data = []
    for year, problems in st.session_state.analyzer.data.items():
        for complexity in ['Low', 'Medium', 'High']:
            count = sum(1 for p in problems if p['complexity'] == complexity)
            complexity_data.append({'Year': year, 'Complexity': complexity, 'Count': count})
    
    if complexity_data:
        df_complex = pd.DataFrame(complexity_data)
        fig = px.bar(df_complex, x='Year', y='Count', color='Complexity',
                    title="Complexity Distribution by Year", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    # Ministry/Organization analysis (for Excel data)
    excel_problems = []
    for problems in st.session_state.analyzer.data.values():
        excel_problems.extend([p for p in problems if p.get('source') == 'excel' and p.get('ministry')])
    
    if excel_problems:
        st.subheader("🏛️ Ministry/Organization Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            ministries = [p['ministry'] for p in excel_problems if p['ministry'] != 'Unknown']
            if ministries:
                ministry_counts = Counter(ministries)
                top_ministries = dict(ministry_counts.most_common(10))
                fig = px.bar(x=list(top_ministries.values()), y=list(top_ministries.keys()), 
                           orientation='h', title="Top 10 Ministries/Organizations")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Type analysis (Software/Hardware)
            types = [p['type'] for p in excel_problems if p['type'] != 'Unknown']
            if types:
                type_counts = Counter(types)
                fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                           title="Problem Types (Software vs Hardware)")
                st.plotly_chart(fig, use_container_width=True)

def show_search_page():
    st.header("🔍 Advanced Problem Search")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload PDF files first.")
        return
    
    # Search filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_query = st.text_input("🔍 Search Keywords:", placeholder="e.g., AI, blockchain, healthcare")
    
    with col2:
        years = list(st.session_state.analyzer.data.keys())
        selected_years = st.multiselect("📅 Select Years:", years, default=years)
    
    with col3:
        all_categories = set()
        for problems in st.session_state.analyzer.data.values():
            all_categories.update(p['category'] for p in problems)
        selected_categories = st.multiselect("🏷️ Categories:", sorted(all_categories))
    
    with col4:
        # Ministry filter for Excel data
        all_ministries = set()
        for problems in st.session_state.analyzer.data.values():
            all_ministries.update(p.get('ministry', 'Unknown') for p in problems if p.get('ministry') and p.get('ministry') != 'Unknown')
        selected_ministries = st.multiselect("🏛️ Ministries:", sorted(all_ministries))
    
    # Additional filters
    col1, col2 = st.columns(2)
    with col1:
        complexity_filter = st.selectbox("⚡ Complexity:", ['All', 'Low', 'Medium', 'High'])
    with col2:
        type_filter = st.selectbox("💻 Type:", ['All', 'Software', 'Hardware'])
    
    # Search and filter
    filtered_problems = []
    for year in selected_years:
        if year in st.session_state.analyzer.data:
            for problem in st.session_state.analyzer.data[year]:
                # Apply filters
                if selected_categories and problem['category'] not in selected_categories:
                    continue
                if selected_ministries and problem.get('ministry', 'Unknown') not in selected_ministries:
                    continue
                if complexity_filter != 'All' and problem['complexity'] != complexity_filter:
                    continue
                if type_filter != 'All' and problem.get('type', 'Unknown') != type_filter:
                    continue
                if search_query:
                    text = (problem['title'] + ' ' + problem['description'] + ' ' + problem.get('ministry', '')).lower()
                    if not any(keyword.lower() in text for keyword in search_query.split()):
                        continue
                
                filtered_problems.append(problem)
    
    # Display results
    st.subheader(f"📋 Search Results ({len(filtered_problems)} problems)")
    
    if filtered_problems:
        for i, problem in enumerate(filtered_problems):
            with st.expander(f"[{problem['year']}] {problem['id']} - {problem['title'][:100]}..."):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Category:** {problem['category']}")
                with col2:
                    st.write(f"**Complexity:** {problem['complexity']}")
                with col3:
                    st.write(f"**Year:** {problem['year']}")
                with col4:
                    if problem.get('type'):
                        st.write(f"**Type:** {problem['type']}")
                
                if problem.get('ministry') and problem['ministry'] != 'Unknown':
                    st.write(f"**Ministry/Organization:** {problem['ministry']}")
                
                st.write("**Description:**")
                st.write(problem['description'][:500] + "..." if len(problem['description']) > 500 else problem['description'])
                
                if problem['keywords']:
                    st.write(f"**Keywords:** {', '.join(problem['keywords'][:10])}")
                
                # Show source
                source_icon = "📊" if problem.get('source') == 'excel' else "📄"
                st.caption(f"{source_icon} Source: {'Excel' if problem.get('source') == 'excel' else 'PDF'}")
    else:
        st.info("No problems match your search criteria.")

def show_trends_page():
    st.header("📈 Trend Analysis")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload PDF files first.")
        return
    
    # Category trends
    st.subheader("🏷️ Category Trends Over Years")
    category_trends = st.session_state.analyzer.get_trending_categories()
    
    if category_trends:
        trend_data = []
        for year, categories in category_trends.items():
            for category, count in categories.items():
                trend_data.append({'Year': year, 'Category': category, 'Count': count})
        
        if trend_data:
            df_trends = pd.DataFrame(trend_data)
            
            # Get top categories
            top_categories = df_trends.groupby('Category')['Count'].sum().nlargest(8).index
            df_filtered = df_trends[df_trends['Category'].isin(top_categories)]
            
            fig = px.line(df_filtered, x='Year', y='Count', color='Category',
                         title="Category Popularity Trends", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # Keyword analysis
    st.subheader("🔤 Trending Keywords")
    all_keywords = []
    for problems in st.session_state.analyzer.data.values():
        for problem in problems:
            all_keywords.extend(problem['keywords'])
    
    if all_keywords:
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(20))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Keywords:**")
            for i, (keyword, count) in enumerate(list(top_keywords.items())[:10]):
                st.write(f"{i+1}. {keyword} ({count})")
        
        with col2:
            # Create word cloud
            if top_keywords:
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(top_keywords)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

def show_recommendations_page():
    st.header("🎯 Personalized Problem Recommendations")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload PDF files first.")
        return
    
    # User input for recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Your Profile")
        interests = st.text_area("🎯 Your Interests (comma-separated):", 
                                placeholder="e.g., AI, healthcare, blockchain, mobile apps")
        skills = st.text_area("🛠️ Your Skills (comma-separated):", 
                             placeholder="e.g., Python, React, machine learning, data analysis")
        
    with col2:
        st.subheader("⚙️ Preferences")
        team_size = st.slider("👥 Team Size:", 1, 8, 4)
        experience_level = st.selectbox("📊 Experience Level:", ['Low', 'Medium', 'High'])
        max_results = st.slider("📋 Max Results:", 5, 50, 15)
    
    if st.button("🚀 Get Recommendations"):
        if interests or skills:
            interest_list = [i.strip() for i in interests.split(',') if i.strip()]
            skill_list = [s.strip() for s in skills.split(',') if s.strip()]
            
            with st.spinner("Analyzing and generating recommendations..."):
                recommendations = st.session_state.analyzer.recommend_problems(
                    interest_list, skill_list, team_size, experience_level
                )
            
            st.subheader(f"🎯 Top {min(max_results, len(recommendations))} Recommendations")
            
            for i, rec in enumerate(recommendations[:max_results]):
                problem = rec['problem']
                score = rec['score']
                reason = rec['match_reason']
                
                with st.expander(f"#{i+1} [{problem['year']}] {problem['id']} - Score: {score}"):
                    st.write(f"**Title:** {problem['title']}")
                    st.write(f"**Category:** {problem['category']} | **Complexity:** {problem['complexity']}")
                    st.write(f"**Why recommended:** {reason}")
                    st.write("**Description:**")
                    st.write(problem['description'][:400] + "..." if len(problem['description']) > 400 else problem['description'])
                    
                    if st.button(f"💾 Save Problem {problem['id']}", key=f"save_{i}"):
                        # Save to session state
                        if 'saved_problems' not in st.session_state:
                            st.session_state.saved_problems = []
                        st.session_state.saved_problems.append(problem)
                        st.success("Problem saved!")
        else:
            st.warning("Please enter your interests or skills to get recommendations.")

def show_comparison_page():
    st.header("📋 Problem Comparison")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload PDF files first.")
        return
    
    # Get all problems for selection
    all_problems = []
    for year, problems in st.session_state.analyzer.data.items():
        for problem in problems:
            problem_display = f"[{problem['year']}] {problem['id']} - {problem['title'][:50]}..."
            all_problems.append((problem_display, problem))
    
    if len(all_problems) < 2:
        st.warning("Need at least 2 problems to compare.")
        return
    
    # Problem selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Select Problems to Compare")
        selected_problems = st.multiselect(
            "Choose problems (max 5):",
            options=[p[0] for p in all_problems],
            max_selections=5
        )
    
    if len(selected_problems) >= 2:
        # Get selected problem objects
        selected_objs = []
        for display_name in selected_problems:
            for p_display, p_obj in all_problems:
                if p_display == display_name:
                    selected_objs.append(p_obj)
                    break
        
        # Comparison table
        st.subheader("📊 Comparison Table")
        comparison_data = []
        for problem in selected_objs:
            comparison_data.append({
                'ID': problem['id'],
                'Year': problem['year'],
                'Category': problem['category'],
                'Complexity': problem['complexity'],
                'Title': problem['title'][:50] + "...",
                'Keywords': ', '.join(problem['keywords'][:5])
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison)
        
        # Similarity analysis
        st.subheader("🔗 Similarity Analysis")
        if len(selected_objs) >= 2:
            similarity_results = []
            for i, prob1 in enumerate(selected_objs):
                similar = st.session_state.analyzer.find_similar_problems(prob1['description'])
                for sim in similar:
                    if sim['problem']['id'] in [p['id'] for p in selected_objs] and sim['problem']['id'] != prob1['id']:
                        similarity_results.append({
                            'Problem 1': prob1['id'],
                            'Problem 2': sim['problem']['id'],
                            'Similarity': f"{sim['similarity']:.2%}"
                        })
            
            if similarity_results:
                df_sim = pd.DataFrame(similarity_results)
                st.dataframe(df_sim)
            else:
                st.info("No significant similarities found between selected problems.")
        
        # Detailed comparison
        st.subheader("📝 Detailed Comparison")
        for i, problem in enumerate(selected_objs):
            with st.expander(f"{problem['id']} - {problem['title'][:100]}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Year:** {problem['year']}")
                    st.write(f"**Category:** {problem['category']}")
                    st.write(f"**Complexity:** {problem['complexity']}")
                with col2:
                    st.write(f"**Keywords:** {', '.join(problem['keywords'][:8])}")
                
                st.write("**Full Description:**")
                st.write(problem['description'])

def show_strategy_generator_page():
    st.header("🎯 Winning Strategy Generator")
    st.markdown("### AI-Powered Hackathon Success Blueprint")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload files first.")
        return
    
    # Get all problems for selection
    all_problems = []
    for year, problems in st.session_state.analyzer.data.items():
        for problem in problems:
            problem_display = f"[{problem['year']}] {problem['id']} - {problem['title'][:50]}..."
            all_problems.append((problem_display, problem))
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        selected_problem_display = st.selectbox(
            "🎯 Select Problem Statement:",
            options=[p[0] for p in all_problems]
        )
    
    with col2:
        generate_btn = st.button("🚀 Generate Winning Strategy", type="primary")
    
    if selected_problem_display and generate_btn:
        # Get selected problem
        selected_problem = None
        for p_display, p_obj in all_problems:
            if p_display == selected_problem_display:
                selected_problem = p_obj
                break
        
        if selected_problem:
            # Team profile collection
            st.subheader("👥 Your Team Profile")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                team_size = st.slider("Team Size:", 2, 6, 4)
            with col2:
                experience = st.selectbox("Experience:", ['Beginner', 'Intermediate', 'Advanced'])
            with col3:
                time_available = st.slider("Days Available:", 1, 30, 14)
            with col4:
                budget = st.selectbox("Budget:", ['Low (₹0-5k)', 'Medium (₹5k-20k)', 'High (₹20k+)'])
            
            # Generate comprehensive analysis
            difficulty = st.session_state.analyzer.calculate_difficulty_score(selected_problem)
            success_prob = st.session_state.analyzer.get_ml_success_prediction(selected_problem)
            
            # Success metrics
            st.subheader("📊 Success Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 AI Success Prediction", f"{success_prob:.1f}%")
            with col2:
                st.metric("⚡ Difficulty Level", f"{difficulty}/10")
            with col3:
                competition_level = "High" if difficulty < 6 else "Medium" if difficulty < 8 else "Low"
                st.metric("🏆 Competition Level", competition_level)
            with col4:
                time_needed = max(7, difficulty * 2)
                st.metric("⏱️ Estimated Days", f"{time_needed}")
            
            # Generate winning strategy
            strategy = st.session_state.analyzer.generate_hackathon_strategy(selected_problem, {
                'size': team_size, 'experience': experience, 'time': time_available
            })
            
            # Strategy visualization
            st.subheader("🗺️ Your Winning Roadmap")
            
            # Timeline visualization
            phases = ['Preparation', 'Development', 'Integration', 'Presentation']
            timeline_days = [
                max(2, time_available * 0.15),  # Preparation
                max(5, time_available * 0.60),  # Development
                max(2, time_available * 0.20),  # Integration
                max(1, time_available * 0.05)   # Presentation
            ]
            
            fig = go.Figure()
            
            cumulative_days = 0
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (phase, days) in enumerate(zip(phases, timeline_days)):
                fig.add_trace(go.Bar(
                    x=[cumulative_days],
                    y=[phase],
                    width=[days],
                    orientation='h',
                    name=f"{phase} ({days:.0f} days)",
                    marker_color=colors[i],
                    text=f"{days:.0f}d",
                    textposition='middle center'
                ))
                cumulative_days += days
            
            fig.update_layout(
                title=f"Project Timeline - {cumulative_days:.0f} Total Days",
                xaxis_title="Days",
                barmode='stack',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed strategy phases
            st.subheader("📋 Detailed Strategy")
            
            # Preparation Phase
            with st.expander("🎯 Phase 1: Preparation & Research", expanded=True):
                st.write("**Duration:** First 15% of timeline")
                for strategy_point in strategy['preparation_phase']:
                    st.write(strategy_point)
                
                st.write("**🔍 Research Checklist:**")
                st.write("• Analyze existing solutions and identify gaps")
                st.write("• Study the problem domain thoroughly")
                st.write("• Identify key technologies and frameworks")
                st.write("• Create technical architecture diagram")
                st.write("• Set up development environment")
            
            # Development Phase
            with st.expander("⚡ Phase 2: Core Development", expanded=True):
                st.write("**Duration:** Main 60% of timeline")
                for strategy_point in strategy['development_phase']:
                    st.write(strategy_point)
                
                # Technology recommendations
                text = (selected_problem['title'] + ' ' + selected_problem['description']).lower()
                st.write("**💻 Recommended Tech Stack:**")
                
                if any(term in text for term in ['ai', 'ml', 'machine learning']):
                    st.write("• **Backend:** Python + FastAPI/Flask")
                    st.write("• **ML:** TensorFlow/PyTorch + scikit-learn")
                    st.write("• **Frontend:** React.js/Streamlit")
                    st.write("• **Database:** PostgreSQL/MongoDB")
                elif any(term in text for term in ['web', 'platform', 'dashboard']):
                    st.write("• **Frontend:** React.js + Material-UI")
                    st.write("• **Backend:** Node.js + Express")
                    st.write("• **Database:** MongoDB/PostgreSQL")
                    st.write("• **Deployment:** Vercel/Netlify")
                elif any(term in text for term in ['mobile', 'app']):
                    st.write("• **Mobile:** React Native/Flutter")
                    st.write("• **Backend:** Firebase/Node.js")
                    st.write("• **Database:** Firebase/MongoDB")
                    st.write("• **APIs:** REST/GraphQL")
                else:
                    st.write("• **Frontend:** React.js")
                    st.write("• **Backend:** Node.js/Python")
                    st.write("• **Database:** PostgreSQL")
                    st.write("• **Cloud:** AWS/Vercel")
            
            # Presentation Phase
            with st.expander("🎭 Phase 3: Presentation & Demo"):
                st.write("**Duration:** Final 5% of timeline")
                for strategy_point in strategy['presentation_phase']:
                    st.write(strategy_point)
                
                st.write("**🎯 Presentation Structure:**")
                st.write("1. **Problem Hook** (30 sec) - Why this matters")
                st.write("2. **Solution Overview** (2 min) - What you built")
                st.write("3. **Live Demo** (3 min) - Show key features")
                st.write("4. **Technical Innovation** (1.5 min) - How it works")
                st.write("5. **Impact & Scalability** (1 min) - Future potential")
                st.write("6. **Q&A Preparation** (2 min) - Handle questions")
            
            # Risk mitigation
            with st.expander("⚠️ Risk Management Plan"):
                for risk_point in strategy['risk_mitigation']:
                    st.write(risk_point)
                
                st.write("**🚨 Critical Backup Plans:**")
                st.write("• Have working MVP 2 days before deadline")
                st.write("• Prepare pre-recorded demo in case of technical issues")
                st.write("• Test presentation setup multiple times")
                st.write("• Assign backup team member for each core component")
            
            # Success factors analysis
            st.subheader("🏆 Success Factors Analysis")
            
            success_factors = {
                'Technical Innovation': 85 if difficulty >= 7 else 70,
                'Market Relevance': 90,
                'Implementation Quality': 75 if experience == 'Advanced' else 60,
                'Presentation Skills': 80,
                'Team Coordination': 85 if team_size <= 4 else 70,
                'Time Management': 90 if time_available >= time_needed else 60
            }
            
            df_factors = pd.DataFrame(list(success_factors.items()), columns=['Factor', 'Score'])
            
            fig = px.bar(df_factors, x='Factor', y='Score', 
                        title="Your Success Factor Analysis",
                        color='Score',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Final recommendations
            st.subheader("🎯 Final Strategic Recommendations")
            
            overall_score = np.mean(list(success_factors.values()))
            
            if overall_score >= 80:
                st.success("🌟 **EXCELLENT MATCH!** You have high chances of success with this problem.")
                st.write("**Focus on:** Technical excellence and innovation showcase")
            elif overall_score >= 65:
                st.warning("⚡ **GOOD POTENTIAL** with proper execution and preparation.")
                st.write("**Focus on:** Solid implementation and clear presentation")
            else:
                st.error("⚠️ **CHALLENGING** - Consider easier problem or more preparation time.")
                st.write("**Focus on:** Basic working solution and learning experience")
            
            # Download strategy report
            if st.button("📥 Download Complete Strategy Report"):
                strategy_report = f"""
# SIH Winning Strategy Report
## Problem: {selected_problem['title']}

### Team Profile
- Size: {team_size} members
- Experience: {experience}
- Timeline: {time_available} days
- Budget: {budget}

### Success Metrics
- AI Prediction: {success_prob:.1f}%
- Difficulty: {difficulty}/10
- Competition: {competition_level}

### Strategy Phases
{json.dumps(strategy, indent=2)}

### Success Factors
{json.dumps(success_factors, indent=2)}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="📄 Download Strategy Report",
                    data=strategy_report,
                    file_name=f"sih_strategy_{selected_problem['id']}.txt",
                    mime="text/plain"
                )

def show_insights_page():
    st.header("💡 Insights Dashboard")
    
    if not st.session_state.analyzer.data:
        st.warning("⚠️ No data loaded. Please upload PDF files first.")
        return
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    total_problems = sum(len(problems) for problems in st.session_state.analyzer.data.values())
    
    # Most popular category
    all_categories = []
    for problems in st.session_state.analyzer.data.values():
        all_categories.extend(p['category'] for p in problems)
    
    if all_categories:
        most_popular_category = Counter(all_categories).most_common(1)[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Most Popular Category", 
                     most_popular_category[0], 
                     f"{most_popular_category[1]} problems")
        
        with col2:
            complexity_counts = Counter()
            for problems in st.session_state.analyzer.data.values():
                complexity_counts.update(p['complexity'] for p in problems)
            most_complex = complexity_counts.most_common(1)[0]
            st.metric("⚡ Most Common Complexity", 
                     most_complex[0], 
                     f"{most_complex[1]} problems")
        
        with col3:
            if len(st.session_state.analyzer.data) > 1:
                years = sorted(st.session_state.analyzer.data.keys())
                growth = len(st.session_state.analyzer.data[years[-1]]) - len(st.session_state.analyzer.data[years[0]])
                st.metric("📈 Growth (Latest vs First)", 
                         f"{growth:+d} problems")
    
    # Recommendations for strategy
    st.subheader("🎯 Strategic Recommendations")
    
    recommendations = []
    
    # Analyze trends
    if len(st.session_state.analyzer.data) > 1:
        # Find growing categories
        category_trends = st.session_state.analyzer.get_trending_categories()
        years = sorted(category_trends.keys())
        if len(years) >= 2:
            latest_year = years[-1]
            prev_year = years[-2]
            
            growth_categories = []
            for category in category_trends[latest_year]:
                current_count = category_trends[latest_year][category]
                prev_count = category_trends.get(prev_year, {}).get(category, 0)
                if current_count > prev_count:
                    growth_categories.append((category, current_count - prev_count))
            
            if growth_categories:
                top_growth = sorted(growth_categories, key=lambda x: x[1], reverse=True)[:3]
                recommendations.append(f"🚀 **Growing categories:** {', '.join([cat for cat, _ in top_growth])}")
    
    # Find underrepresented but important areas
    if all_categories:
        category_counts = Counter(all_categories)
        important_keywords = ['ai', 'machine learning', 'blockchain', 'sustainability', 'healthcare']
        
        underrep_important = []
        for keyword in important_keywords:
            count = sum(1 for cat in category_counts if keyword.lower() in cat.lower())
            if count < total_problems * 0.1:  # Less than 10% representation
                underrep_important.append(keyword)
        
        if underrep_important:
            recommendations.append(f"💡 **Emerging opportunities:** {', '.join(underrep_important)}")
    
    # Complexity recommendations
    complexity_counts = Counter()
    for problems in st.session_state.analyzer.data.values():
        complexity_counts.update(p['complexity'] for p in problems)
    
    if complexity_counts['High'] < total_problems * 0.3:
        recommendations.append("⚡ **Consider high-complexity problems** for better learning and impact")
    
    if recommendations:
        for rec in recommendations:
            st.success(rec)
    else:
        st.info("Upload more data to get strategic insights.")
    
    # Saved problems
    if 'saved_problems' in st.session_state and st.session_state.saved_problems:
        st.subheader("💾 Your Saved Problems")
        for i, problem in enumerate(st.session_state.saved_problems):
            with st.expander(f"Saved #{i+1}: {problem['id']} - {problem['title'][:50]}..."):
                st.write(f"**Category:** {problem['category']} | **Complexity:** {problem['complexity']}")
                st.write(f"**Year:** {problem['year']}")
                st.write(problem['description'][:300] + "...")

if __name__ == "__main__":
    main()
