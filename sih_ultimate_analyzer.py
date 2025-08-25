import streamlit as st
import pandas as pd
import PyPDF2
import os
import re
import json
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import nltk
from textblob import TextBlob
from fuzzywuzzy import fuzz, process
import numpy as np
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Advanced imports for new features
import requests
from urllib.parse import quote
import hashlib
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedSIHAnalyzer:
    def __init__(self):
        self.data = {}
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.problem_vectors = None
        self.success_predictor = None
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
        
    def analyze_market_demand(self, category):
        """Simulate market demand analysis for a category"""
        # In real implementation, this could connect to job APIs, trend APIs, etc.
        demand_scores = {
            'AI/ML': 95, 'Blockchain': 78, 'IoT': 85, 'Cloud': 92, 'Cybersecurity': 88,
            'Data Science': 90, 'Mobile': 82, 'Web': 75, 'AR/VR': 70, 'Robotics': 80,
            'Healthcare': 85, 'Agriculture': 70, 'Education': 75, 'Smart Cities': 80,
            'Environment': 85, 'Fintech': 88, 'Other': 60
        }
        return demand_scores.get(category, 60)
    
    def calculate_difficulty_score(self, problem):
        """Calculate comprehensive difficulty score"""
        score = 0
        text = (problem['title'] + ' ' + problem['description']).lower()
        
        # Technical complexity indicators
        high_complexity = ['machine learning', 'blockchain', 'ai', 'neural network', 'deep learning', 
                          'real-time', 'scalable', 'distributed', 'microservices', 'cloud native']
        medium_complexity = ['api', 'database', 'web application', 'mobile app', 'dashboard', 
                            'integration', 'automation', 'analytics']
        low_complexity = ['website', 'form', 'simple', 'basic', 'static', 'display', 'report']
        
        for term in high_complexity:
            if term in text:
                score += 3
        for term in medium_complexity:
            if term in text:
                score += 2
        for term in low_complexity:
            if term in text:
                score += 1
                
        # Normalize to 1-10 scale
        return min(10, max(1, score))
    
    def predict_success_probability(self, problem, team_skills, team_experience):
        """Predict success probability using multiple factors"""
        base_score = 50  # Base 50% chance
        
        # Skill matching bonus
        skill_match = 0
        problem_text = (problem['title'] + ' ' + problem['description']).lower()
        for skill in team_skills:
            if skill.lower() in problem_text:
                skill_match += 15
        
        # Experience vs difficulty adjustment
        difficulty = self.calculate_difficulty_score(problem)
        if team_experience == 'High' and difficulty <= 7:
            experience_bonus = 20
        elif team_experience == 'Medium' and 4 <= difficulty <= 8:
            experience_bonus = 15
        elif team_experience == 'Low' and difficulty <= 5:
            experience_bonus = 10
        else:
            experience_bonus = -10 if difficulty > 8 else 0
        
        # Market demand bonus
        market_demand = self.analyze_market_demand(problem['category'])
        demand_bonus = (market_demand - 70) / 10 * 5  # Scale to bonus points
        
        # Trending technology bonus
        trend_bonus = 0
        for tech, keywords in self.tech_trends.items():
            if any(keyword in problem_text for keyword in keywords):
                trend_bonus += 8
                break
        
        total_score = base_score + skill_match + experience_bonus + demand_bonus + trend_bonus
        return min(95, max(5, total_score))
    
    def generate_project_timeline(self, problem, team_size=4):
        """Generate realistic project timeline"""
        difficulty = self.calculate_difficulty_score(problem)
        base_weeks = {1: 12, 2: 10, 3: 8, 4: 6, 5: 5, 6: 4}  # Base weeks by difficulty level
        
        weeks = base_weeks.get(min(6, max(1, difficulty // 2 + 1)), 8)
        weeks = max(3, weeks - (team_size - 4))  # Adjust for team size
        
        phases = [
            {'phase': 'Research & Planning', 'weeks': max(1, weeks * 0.2), 'tasks': ['Problem analysis', 'Tech stack selection', 'Architecture design']},
            {'phase': 'Development Setup', 'weeks': max(1, weeks * 0.15), 'tasks': ['Environment setup', 'Repository creation', 'Initial prototyping']},
            {'phase': 'Core Development', 'weeks': max(2, weeks * 0.45), 'tasks': ['Feature implementation', 'Backend development', 'Frontend creation']},
            {'phase': 'Integration & Testing', 'weeks': max(1, weeks * 0.15), 'tasks': ['Component integration', 'Testing', 'Bug fixes']},
            {'phase': 'Final Polish', 'weeks': max(1, weeks * 0.05), 'tasks': ['Documentation', 'Presentation prep', 'Final testing']}
        ]
        
        return phases, weeks
    
    def suggest_tech_stack(self, problem):
        """Suggest optimal technology stack"""
        text = (problem['title'] + ' ' + problem['description']).lower()
        suggestions = {
            'frontend': [], 'backend': [], 'database': [], 'tools': [], 'cloud': []
        }
        
        # AI/ML problems
        if any(term in text for term in ['ai', 'ml', 'machine learning', 'neural', 'prediction']):
            suggestions['backend'].extend(['Python', 'TensorFlow/PyTorch', 'FastAPI'])
            suggestions['tools'].extend(['Jupyter', 'scikit-learn', 'pandas'])
            suggestions['cloud'].append('AWS SageMaker / Google AI Platform')
        
        # Web applications
        if any(term in text for term in ['web', 'platform', 'portal', 'dashboard']):
            suggestions['frontend'].extend(['React.js', 'Vue.js', 'Angular'])
            suggestions['backend'].extend(['Node.js', 'Express.js', 'Django'])
            
        # Mobile applications
        if any(term in text for term in ['mobile', 'app', 'android', 'ios']):
            suggestions['frontend'].extend(['React Native', 'Flutter', 'Ionic'])
            
        # Database suggestions
        if any(term in text for term in ['data', 'records', 'information', 'storage']):
            suggestions['database'].extend(['PostgreSQL', 'MongoDB', 'Redis'])
            
        # Blockchain
        if any(term in text for term in ['blockchain', 'smart contract', 'crypto']):
            suggestions['backend'].extend(['Solidity', 'Web3.js', 'Ethereum'])
            
        # Default fallbacks
        if not suggestions['frontend']:
            suggestions['frontend'].append('React.js')
        if not suggestions['backend']:
            suggestions['backend'].append('Node.js')
        if not suggestions['database']:
            suggestions['database'].append('PostgreSQL')
            
        return suggestions
    
    def calculate_resource_requirements(self, problem, team_size=4):
        """Calculate detailed resource requirements"""
        difficulty = self.calculate_difficulty_score(problem)
        
        # Budget estimation (in thousands)
        base_budget = difficulty * 2
        cloud_cost = base_budget * 0.3
        tools_cost = base_budget * 0.2
        misc_cost = base_budget * 0.1
        
        # Skills needed
        text = (problem['title'] + ' ' + problem['description']).lower()
        required_skills = []
        
        skill_mapping = {
            'Frontend Developer': ['web', 'frontend', 'ui', 'ux', 'dashboard', 'interface'],
            'Backend Developer': ['backend', 'api', 'server', 'database'],
            'AI/ML Engineer': ['ai', 'ml', 'machine learning', 'neural', 'prediction'],
            'Mobile Developer': ['mobile', 'android', 'ios', 'app'],
            'Data Scientist': ['data', 'analytics', 'visualization', 'insights'],
            'DevOps Engineer': ['cloud', 'deployment', 'scalable', 'infrastructure'],
            'Security Expert': ['security', 'encryption', 'authentication', 'cyber'],
            'Blockchain Developer': ['blockchain', 'smart contract', 'crypto']
        }
        
        for skill, keywords in skill_mapping.items():
            if any(keyword in text for keyword in keywords):
                required_skills.append(skill)
        
        if not required_skills:
            required_skills = ['Full Stack Developer', 'UI/UX Designer']
            
        return {
            'budget_estimate': {
                'total': base_budget,
                'cloud_services': cloud_cost,
                'tools_software': tools_cost,
                'miscellaneous': misc_cost
            },
            'required_skills': required_skills[:team_size],
            'time_commitment': f"{difficulty * 10}-{difficulty * 15} hours/week per person"
        }
    
    def analyze_competition_level(self, problem):
        """Analyze expected competition level"""
        category_popularity = {
            'AI/ML': 90, 'Blockchain': 70, 'Healthcare': 85, 'Education': 75,
            'Smart Cities': 80, 'Agriculture': 65, 'Environment': 70, 'Fintech': 85,
            'IoT': 75, 'Cybersecurity': 80, 'Other': 50
        }
        
        popularity = category_popularity.get(problem['category'], 60)
        difficulty = self.calculate_difficulty_score(problem)
        
        # High popularity + low difficulty = high competition
        # Low popularity + high difficulty = low competition
        competition_score = popularity - (difficulty * 5)
        
        if competition_score >= 70:
            return {'level': 'High', 'expected_teams': '50-100+', 'advice': 'Need exceptional execution'}
        elif competition_score >= 40:
            return {'level': 'Medium', 'expected_teams': '20-50', 'advice': 'Good opportunity with solid execution'}
        else:
            return {'level': 'Low', 'expected_teams': '5-20', 'advice': 'Great opportunity for innovative solutions'}
    
    def generate_learning_resources(self, problem):
        """Generate learning resources and preparation materials"""
        text = (problem['title'] + ' ' + problem['description']).lower()
        resources = {
            'courses': [], 'documentation': [], 'tutorials': [], 'books': [], 'tools': []
        }
        
        # AI/ML resources
        if any(term in text for term in ['ai', 'ml', 'machine learning', 'neural']):
            resources['courses'].extend([
                'Machine Learning by Andrew Ng (Coursera)',
                'Deep Learning Specialization (Coursera)',
                'Fast.ai Practical Deep Learning'
            ])
            resources['documentation'].extend([
                'TensorFlow Documentation',
                'PyTorch Tutorials',
                'scikit-learn User Guide'
            ])
            
        # Web development
        if any(term in text for term in ['web', 'platform', 'dashboard']):
            resources['courses'].extend([
                'React - The Complete Guide (Udemy)',
                'Node.js Developer Course',
                'Full Stack Web Development'
            ])
            
        # Blockchain
        if any(term in text for term in ['blockchain', 'smart contract']):
            resources['courses'].extend([
                'Blockchain Specialization (Coursera)',
                'Ethereum and Solidity Course',
                'Cryptocurrency and Blockchain Course'
            ])
            
        return resources
    
    def create_risk_assessment(self, problem, team_profile):
        """Create comprehensive risk assessment"""
        risks = []
        
        difficulty = self.calculate_difficulty_score(problem)
        if difficulty > 8:
            risks.append({
                'type': 'Technical Complexity',
                'level': 'High',
                'description': 'Problem involves advanced technical concepts',
                'mitigation': 'Invest extra time in learning and prototyping'
            })
        
        # Market demand analysis
        market_demand = self.analyze_market_demand(problem['category'])
        if market_demand < 70:
            risks.append({
                'type': 'Market Relevance',
                'level': 'Medium',
                'description': 'Limited industry demand for this category',
                'mitigation': 'Focus on innovative approach and strong presentation'
            })
        
        # Competition analysis
        competition = self.analyze_competition_level(problem)
        if competition['level'] == 'High':
            risks.append({
                'type': 'High Competition',
                'level': 'High',
                'description': f"Expected {competition['expected_teams']} competing teams",
                'mitigation': 'Develop unique value proposition and exceptional execution'
            })
        
        return risks

# Initialize the advanced analyzer
if 'advanced_analyzer' not in st.session_state:
    st.session_state.advanced_analyzer = AdvancedSIHAnalyzer()

def show_advanced_analysis_page():
    st.header("🚀 Advanced Problem Analysis")
    
    if not st.session_state.advanced_analyzer.data:
        st.warning("⚠️ No data loaded. Please upload files first.")
        return
    
    # Get all problems for selection
    all_problems = []
    for year, problems in st.session_state.advanced_analyzer.data.items():
        for problem in problems:
            problem_display = f"[{problem['year']}] {problem['id']} - {problem['title'][:50]}..."
            all_problems.append((problem_display, problem))
    
    selected_problem_display = st.selectbox(
        "🎯 Select a problem for detailed analysis:",
        options=[p[0] for p in all_problems]
    )
    
    if selected_problem_display:
        # Get selected problem object
        selected_problem = None
        for p_display, p_obj in all_problems:
            if p_display == selected_problem_display:
                selected_problem = p_obj
                break
        
        if selected_problem:
            # Team profile input
            st.subheader("👥 Your Team Profile")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                team_size = st.slider("Team Size:", 2, 6, 4)
                team_experience = st.selectbox("Team Experience:", ['Low', 'Medium', 'High'])
            
            with col2:
                team_skills = st.text_area("Team Skills (comma-separated):", 
                                         placeholder="Python, React, Machine Learning, etc.")
            
            with col3:
                available_time = st.slider("Available Hours/Week:", 10, 40, 20)
            
            if st.button("🔍 Generate Advanced Analysis"):
                skills_list = [s.strip() for s in team_skills.split(',') if s.strip()]
                
                # Generate comprehensive analysis
                st.subheader("📊 Comprehensive Analysis Report")
                
                # Success Probability
                success_prob = st.session_state.advanced_analyzer.predict_success_probability(
                    selected_problem, skills_list, team_experience
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Success Probability", f"{success_prob:.1f}%")
                with col2:
                    difficulty = st.session_state.advanced_analyzer.calculate_difficulty_score(selected_problem)
                    st.metric("⚡ Difficulty Score", f"{difficulty}/10")
                with col3:
                    market_demand = st.session_state.advanced_analyzer.analyze_market_demand(selected_problem['category'])
                    st.metric("📈 Market Demand", f"{market_demand}/100")
                with col4:
                    competition = st.session_state.advanced_analyzer.analyze_competition_level(selected_problem)
                    st.metric("🏆 Competition", competition['level'])
                
                # Timeline Analysis
                st.subheader("📅 Project Timeline")
                phases, total_weeks = st.session_state.advanced_analyzer.generate_project_timeline(
                    selected_problem, team_size
                )
                
                timeline_data = []
                start_week = 0
                for phase in phases:
                    timeline_data.append({
                        'Phase': phase['phase'],
                        'Duration (weeks)': phase['weeks'],
                        'Start Week': start_week,
                        'End Week': start_week + phase['weeks'],
                        'Tasks': ', '.join(phase['tasks'])
                    })
                    start_week += phase['weeks']
                
                df_timeline = pd.DataFrame(timeline_data)
                
                # Create Gantt chart
                fig = px.timeline(df_timeline, x_start='Start Week', x_end='End Week', y='Phase',
                                color='Phase', title=f"Project Timeline ({total_weeks:.1f} weeks total)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Technology Stack Suggestions
                st.subheader("💻 Recommended Technology Stack")
                tech_stack = st.session_state.advanced_analyzer.suggest_tech_stack(selected_problem)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Frontend:**")
                    for tech in tech_stack['frontend']:
                        st.write(f"• {tech}")
                    
                with col2:
                    st.write("**Backend:**")
                    for tech in tech_stack['backend']:
                        st.write(f"• {tech}")
                    if tech_stack['database']:
                        st.write("**Database:**")
                        for tech in tech_stack['database']:
                            st.write(f"• {tech}")
                
                with col3:
                    if tech_stack['tools']:
                        st.write("**Tools:**")
                        for tech in tech_stack['tools']:
                            st.write(f"• {tech}")
                    if tech_stack['cloud']:
                        st.write("**Cloud:**")
                        for tech in tech_stack['cloud']:
                            st.write(f"• {tech}")
                
                # Resource Requirements
                st.subheader("💰 Resource Requirements")
                resources = st.session_state.advanced_analyzer.calculate_resource_requirements(
                    selected_problem, team_size
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Budget Estimate (₹):**")
                    budget = resources['budget_estimate']
                    st.write(f"• Total: ₹{budget['total']*1000:,.0f}")
                    st.write(f"• Cloud Services: ₹{budget['cloud_services']*1000:,.0f}")
                    st.write(f"• Tools/Software: ₹{budget['tools_software']*1000:,.0f}")
                    st.write(f"• Miscellaneous: ₹{budget['miscellaneous']*1000:,.0f}")
                
                with col2:
                    st.write("**Required Skills:**")
                    for skill in resources['required_skills']:
                        st.write(f"• {skill}")
                    st.write(f"**Time Commitment:** {resources['time_commitment']}")
                
                # Risk Assessment
                st.subheader("⚠️ Risk Assessment")
                risks = st.session_state.advanced_analyzer.create_risk_assessment(
                    selected_problem, {'size': team_size, 'experience': team_experience}
                )
                
                for risk in risks:
                    color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}[risk['level']]
                    st.error(f"**{risk['type']}** ({risk['level']} Risk)")
                    st.write(f"Description: {risk['description']}")
                    st.write(f"Mitigation: {risk['mitigation']}")
                    st.write("---")
                
                # Learning Resources
                st.subheader("📚 Learning Resources")
                learning_resources = st.session_state.advanced_analyzer.generate_learning_resources(selected_problem)
                
                if learning_resources['courses']:
                    st.write("**Recommended Courses:**")
                    for course in learning_resources['courses'][:5]:
                        st.write(f"• {course}")
                
                # Final Recommendation
                st.subheader("🎯 Final Recommendation")
                
                if success_prob >= 70:
                    st.success(f"✅ **HIGHLY RECOMMENDED** - This problem is an excellent match for your team!")
                elif success_prob >= 50:
                    st.warning(f"⚠️ **MODERATE RECOMMENDATION** - Good potential with proper preparation")
                else:
                    st.error(f"❌ **NOT RECOMMENDED** - Consider problems better suited to your current capabilities")
                
                st.write(f"**Overall Assessment:** {competition['advice']}")

def show_ai_insights_page():
    st.header("🤖 AI-Powered Insights")
    
    if not st.session_state.advanced_analyzer.data:
        st.warning("⚠️ No data loaded. Please upload files first.")
        return
    
    # Technology trend analysis
    st.subheader("📈 Technology Trend Analysis")
    
    # Analyze technology mentions across years
    tech_trends_data = []
    for year, problems in st.session_state.advanced_analyzer.data.items():
        year_tech_counts = defaultdict(int)
        for problem in problems:
            text = (problem['title'] + ' ' + problem['description']).lower()
            for tech, keywords in st.session_state.advanced_analyzer.tech_trends.items():
                if any(keyword in text for keyword in keywords):
                    year_tech_counts[tech] += 1
        
        for tech, count in year_tech_counts.items():
            tech_trends_data.append({'Year': year, 'Technology': tech, 'Mentions': count})
    
    if tech_trends_data:
        df_tech_trends = pd.DataFrame(tech_trends_data)
        fig = px.line(df_tech_trends, x='Year', y='Mentions', color='Technology',
                     title="Technology Trends Over Years", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Success probability distribution
    st.subheader("🎯 Success Probability Analysis")
    
    # Generate success probabilities for all problems
    sample_skills = ['Python', 'JavaScript', 'React', 'Machine Learning', 'Database']
    success_data = []
    
    for year, problems in st.session_state.advanced_analyzer.data.items():
        for problem in problems[:10]:  # Sample to avoid too much computation
            success_prob = st.session_state.advanced_analyzer.predict_success_probability(
                problem, sample_skills, 'Medium'
            )
            success_data.append({
                'Problem': problem['id'],
                'Category': problem['category'],
                'Success Probability': success_prob,
                'Year': year
            })
    
    if success_data:
        df_success = pd.DataFrame(success_data)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_success, x='Success Probability', nbins=20,
                             title="Success Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_success_by_category = df_success.groupby('Category')['Success Probability'].mean().reset_index()
            fig = px.bar(avg_success_by_category, x='Category', y='Success Probability',
                        title="Average Success Probability by Category")
            st.plotly_chart(fig, use_container_width=True)

def show_team_optimizer_page():
    st.header("👥 Team Optimization Assistant")
    
    st.subheader("🎯 Find Your Perfect Problem Match")
    
    # Team profile builder
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Details")
        team_size = st.slider("Team Size:", 2, 6, 4)
        team_experience = st.selectbox("Overall Experience Level:", ['Beginner', 'Intermediate', 'Advanced'])
        available_hours = st.slider("Available Hours per Week:", 10, 50, 25)
        
    with col2:
        st.subheader("Skills & Interests")
        programming_languages = st.multiselect(
            "Programming Languages:",
            ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust', 'PHP', 'Swift', 'Kotlin']
        )
        
        frameworks = st.multiselect(
            "Frameworks & Technologies:",
            ['React', 'Angular', 'Vue.js', 'Django', 'Flask', 'Node.js', 'TensorFlow', 'PyTorch', 'Docker']
        )
        
        domains = st.multiselect(
            "Domain Interests:",
            ['AI/ML', 'Web Development', 'Mobile Apps', 'Blockchain', 'IoT', 'Cybersecurity', 'Data Science']
        )
    
    if st.button("🔍 Find Optimal Problem Matches"):
        if not st.session_state.advanced_analyzer.data:
            st.warning("Please upload data first!")
            return
            
        # Analyze all problems for this team
        recommendations = []
        all_skills = programming_languages + frameworks + domains
        
        for year, problems in st.session_state.advanced_analyzer.data.items():
            for problem in problems:
                success_prob = st.session_state.advanced_analyzer.predict_success_probability(
                    problem, all_skills, team_experience
                )
                
                difficulty = st.session_state.advanced_analyzer.calculate_difficulty_score(problem)
                market_demand = st.session_state.advanced_analyzer.analyze_market_demand(problem['category'])
                competition = st.session_state.advanced_analyzer.analyze_competition_level(problem)
                
                # Calculate overall score
                overall_score = (success_prob * 0.4 + market_demand * 0.3 + 
                               (100 - competition['level'] == 'High' * 30) * 0.3)
                
                recommendations.append({
                    'problem': problem,
                    'success_probability': success_prob,
                    'difficulty': difficulty,
                    'market_demand': market_demand,
                    'competition_level': competition['level'],
                    'overall_score': overall_score
                })
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        st.subheader("🏆 Top Recommendations for Your Team")
        
        for i, rec in enumerate(recommendations[:10]):
            problem = rec['problem']
            with st.expander(f"#{i+1} [{problem['year']}] {problem['id']} - Score: {rec['overall_score']:.1f}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Success Probability", f"{rec['success_probability']:.1f}%")
                    st.metric("Difficulty", f"{rec['difficulty']}/10")
                
                with col2:
                    st.metric("Market Demand", f"{rec['market_demand']}/100")
                    st.metric("Competition", rec['competition_level'])
                
                with col3:
                    # Generate quick timeline
                    phases, weeks = st.session_state.advanced_analyzer.generate_project_timeline(problem, team_size)
                    st.metric("Estimated Timeline", f"{weeks:.0f} weeks")
                    st.metric("Weekly Commitment", f"{(weeks * 40 / available_hours):.1f}x available time")
                
                st.write(f"**Category:** {problem['category']}")
                st.write(f"**Title:** {problem['title']}")
                st.write(f"**Description:** {problem['description'][:200]}...")
                
                # Quick tech stack preview
                tech_stack = st.session_state.advanced_analyzer.suggest_tech_stack(problem)
                st.write(f"**Suggested Tech:** {', '.join(tech_stack['backend'][:2] + tech_stack['frontend'][:2])}")

# Update main function to include new pages
def main():
    st.set_page_config(
        page_title="🚀 ULTIMATE SIH Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 ULTIMATE SIH Problem Statement Analyzer")
    st.markdown("### AI-Powered Multi-Year Analysis & Success Prediction System")
    
    # Copy data from basic analyzer to advanced analyzer
    if hasattr(st.session_state, 'analyzer') and st.session_state.analyzer.data:
        st.session_state.advanced_analyzer.data = st.session_state.analyzer.data
    
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
            "🚀 Advanced Analysis",  # New
            "🤖 AI Insights",       # New
            "👥 Team Optimizer",    # New
            "💡 Insights Dashboard"
        ])
    
    # Route to appropriate page
    if page == "🚀 Advanced Analysis":
        show_advanced_analysis_page()
    elif page == "🤖 AI Insights":
        show_ai_insights_page()
    elif page == "👥 Team Optimizer":
        show_team_optimizer_page()
    else:
        # Use existing functions from the original analyzer
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
        elif page == "💡 Insights Dashboard":
            show_insights_page()

if __name__ == "__main__":
    main()
