import streamlit as st
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Medical Prescription Verification",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class DrugInfo:
    name: str
    dosage: str
    frequency: str
    route: str = "oral"

@dataclass
class AnalysisResult:
    interactions: List[str]
    dosage_recommendations: List[str]
    alternatives: List[str]
    warnings: List[str]
    safety_score: float

class IBMGraniteAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_error = None
    
    @st.cache_resource
    def load_model(_self):
        """Load IBM Granite model with caching"""
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing tokenizer...")
            progress_bar.progress(25)
            _self.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
            
            status_text.text("🔄 Loading model (this may take 5-10 minutes for first download)...")
            progress_bar.progress(50)
            _self.model = AutoModelForCausalLM.from_pretrained(
                "ibm-granite/granite-3.3-2b-instruct",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            status_text.text("🔄 Finalizing setup...")
            progress_bar.progress(100)
            _self.is_loaded = True
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            st.success("✅ IBM Granite model loaded successfully!")
            return _self.tokenizer, _self.model
        except Exception as e:
            _self.load_error = str(e)
            st.error(f"❌ Error loading model: {str(e)}")
            st.error("💡 Try refreshing the page or check your internet connection")
            return None, None
    
    def extract_drug_info(self, text: str) -> List[DrugInfo]:
        """Extract structured drug information from unstructured text"""
        if not self.is_loaded:
            self.tokenizer, self.model = self.load_model()
        
        if not self.model:
            return []
        
        messages = [
            {"role": "user", "content": f"""
            Extract drug information from the following prescription text and format as JSON:
            Text: "{text}"
            
            Please extract and return ONLY a valid JSON array with this exact structure:
            [
                {{
                    "name": "drug_name",
                    "dosage": "amount_with_unit",
                    "frequency": "how_often",
                    "route": "administration_route"
                }}
            ]
            
            Return only the JSON array, no other text.
            """}
        ]
        
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                drug_data = json.loads(json_str)
                return [DrugInfo(**drug) for drug in drug_data]
            
        except Exception as e:
            st.warning(f"Error extracting drug info: {str(e)}")
        
        return []
    
    def analyze_prescription(self, drugs: List[DrugInfo], age: int) -> AnalysisResult:
        """Analyze prescription for interactions, dosages, and alternatives"""
        if not self.is_loaded:
            self.tokenizer, self.model = self.load_model()
        
        if not self.model:
            return AnalysisResult([], [], [], [], 0.0)
        
        drug_list = ", ".join([f"{drug.name} {drug.dosage} {drug.frequency}" for drug in drugs])
        
        messages = [
            {"role": "user", "content": f"""
            As a medical AI assistant, analyze this prescription for a {age}-year-old patient:
            Drugs: {drug_list}
            
            Provide analysis in this exact JSON format:
            {{
                "interactions": ["list of drug interactions with severity"],
                "dosage_recommendations": ["age-specific dosage recommendations"],
                "alternatives": ["safer alternative medications if needed"],
                "warnings": ["important warnings and contraindications"],
                "safety_score": 0.85
            }}
            
            Focus on:
            1. Drug-drug interactions and their clinical significance
            2. Age-appropriate dosing for {age}-year-old patient
            3. Safer alternatives if current drugs pose risks
            4. Important warnings based on age and drug combination
            5. Overall safety score (0-1 scale)
            
            Return only valid JSON, no other text.
            """}
        ]
        
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis_data = json.loads(json_str)
                return AnalysisResult(
                    interactions=analysis_data.get("interactions", []),
                    dosage_recommendations=analysis_data.get("dosage_recommendations", []),
                    alternatives=analysis_data.get("alternatives", []),
                    warnings=analysis_data.get("warnings", []),
                    safety_score=analysis_data.get("safety_score", 0.5)
                )
            
        except Exception as e:
            st.warning(f"Error analyzing prescription: {str(e)}")
        
        return AnalysisResult(
            interactions=["Analysis temporarily unavailable"],
            dosage_recommendations=["Please consult healthcare provider"],
            alternatives=["Consult pharmacist for alternatives"],
            warnings=["Manual review recommended"],
            safety_score=0.5
        )

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .safety-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .safe { background-color: #D5F4E6; color: #27AE60; }
    .moderate { background-color: #FCF3CF; color: #F39C12; }
    .unsafe { background-color: #FADBD8; color: #E74C3C; }
    .drug-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498DB;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #FEF9E7;
        border: 1px solid #F4D03F;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .loading-info {
        background-color: #E8F6FF;
        border: 1px solid #3498DB;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">💊 AI Medical Prescription Verification</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by IBM Granite Model | Drug Interaction Analysis & Safety Verification**")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = IBMGraniteAnalyzer()
    
    # Check if model is loading for the first time
    if not st.session_state.analyzer.is_loaded and st.session_state.analyzer.load_error is None:
        st.markdown("""
        <div class="loading-info">
        <h3>🚀 Welcome to AI Medical Prescription Verification!</h3>
        <p><strong>First-time setup:</strong> The IBM Granite model (~4.6GB) will be downloaded automatically.</p>
        <p>This process takes 5-10 minutes depending on your internet speed.</p>
        <p>The model will be cached locally for faster future use.</p>
        <br>
        <p><em>Click "Initialize System" below to begin the download process.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔥 Initialize System", type="primary"):
            st.session_state.analyzer.load_model()
            st.rerun()
        
        return
    
    # Show error message if model failed to load
    if st.session_state.analyzer.load_error:
        st.error("❌ Model loading failed. Please try the following:")
        st.write("1. Check your internet connection")
        st.write("2. Restart the application")
        st.write("3. Clear browser cache and refresh")
        if st.button("🔄 Retry Loading"):
            st.session_state.analyzer.load_error = None
            st.session_state.analyzer.is_loaded = False
            st.rerun()
        return
    
    # Sidebar for patient information
    with st.sidebar:
        st.header("📋 Patient Information")
        age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
        
        st.header("⚙️ Analysis Options")
        auto_analyze = st.checkbox("Auto-analyze on input change", value=True)
        
        st.header("ℹ️ About")
        st.info("""
        This AI system uses IBM Granite model to:
        • Extract drug information from text
        • Detect harmful drug interactions
        • Provide age-specific dosage recommendations
        • Suggest safer alternatives
        • Generate comprehensive safety analysis
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Prescription Input")
        
        input_method = st.radio("Choose input method:", ["Manual Entry", "Text Upload"])
        
        drugs = []
        
        if input_method == "Manual Entry":
            st.subheader("Enter Drug Details")
            
            # Dynamic drug entry
            if 'drug_count' not in st.session_state:
                st.session_state.drug_count = 1
            
            for i in range(st.session_state.drug_count):
                with st.expander(f"Drug {i+1}", expanded=True):
                    col_name, col_dose = st.columns(2)
                    with col_name:
                        drug_name = st.text_input(f"Drug Name {i+1}", key=f"drug_name_{i}")
                    with col_dose:
                        dosage = st.text_input(f"Dosage {i+1}", placeholder="e.g., 500mg", key=f"dosage_{i}")
                    
                    col_freq, col_route = st.columns(2)
                    with col_freq:
                        frequency = st.selectbox(f"Frequency {i+1}", 
                                               ["Once daily", "Twice daily", "Three times daily", "Four times daily", "As needed"], 
                                               key=f"frequency_{i}")
                    with col_route:
                        route = st.selectbox(f"Route {i+1}", ["Oral", "Topical", "Injectable", "Inhalation"], key=f"route_{i}")
                    
                    if drug_name:
                        drugs.append(DrugInfo(drug_name, dosage, frequency, route))
            
            col_add, col_remove = st.columns(2)
            with col_add:
                if st.button("➕ Add Drug"):
                    st.session_state.drug_count += 1
                    st.rerun()
            with col_remove:
                if st.button("➖ Remove Drug") and st.session_state.drug_count > 1:
                    st.session_state.drug_count -= 1
                    st.rerun()
        
        else:  # Text Upload
            st.subheader("Upload Prescription Text")
            prescription_text = st.text_area(
                "Paste prescription details here:",
                placeholder="e.g., Patient prescribed Paracetamol 500mg twice daily and Ibuprofen 200mg as needed for pain...",
                height=150
            )
            
            if prescription_text and st.button("🔍 Extract Drug Information"):
                with st.spinner("Extracting drug information using IBM Granite..."):
                    drugs = st.session_state.analyzer.extract_drug_info(prescription_text)
                
                if drugs:
                    st.success(f"✅ Extracted {len(drugs)} drug(s) from text")
                    for drug in drugs:
                        st.markdown(f"""
                        <div class="drug-card">
                        <strong>{drug.name}</strong> - {drug.dosage} {drug.frequency} ({drug.route})
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No drugs extracted. Please try manual entry.")
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if drugs and (auto_analyze or st.button("🚀 Analyze Prescription", type="primary")):
            with st.spinner("Analyzing prescription with IBM Granite model..."):
                analysis = st.session_state.analyzer.analyze_prescription(drugs, age)
            
            # Safety Score Display
            safety_class = "safe" if analysis.safety_score >= 0.8 else "moderate" if analysis.safety_score >= 0.6 else "unsafe"
            st.markdown(f"""
            <div class="safety-score {safety_class}">
            Safety Score: {analysis.safety_score:.1%}
            </div>
            """, unsafe_allow_html=True)
            
            # Drug Interactions
            with st.expander("⚠️ Drug Interactions", expanded=True):
                if analysis.interactions:
                    for interaction in analysis.interactions:
                        st.warning(f"🔄 {interaction}")
                else:
                    st.success("✅ No significant drug interactions detected")
            
            # Dosage Recommendations
            with st.expander("📏 Dosage Recommendations", expanded=True):
                if analysis.dosage_recommendations:
                    for recommendation in analysis.dosage_recommendations:
                        st.info(f"💊 {recommendation}")
                else:
                    st.success("✅ Current dosages appear appropriate")
            
            # Alternative Medications
            with st.expander("🔄 Alternative Medications", expanded=True):
                if analysis.alternatives:
                    for alternative in analysis.alternatives:
                        st.info(f"💡 {alternative}")
                else:
                    st.success("✅ Current medications are optimal")
            
            # Warnings
            with st.expander("⚠️ Important Warnings", expanded=True):
                if analysis.warnings:
                    for warning in analysis.warnings:
                        st.error(f"⚠️ {warning}")
                else:
                    st.success("✅ No critical warnings")
        
        elif not drugs:
            st.info("👆 Enter prescription details above to start analysis")
        
        # Quick Actions
        if drugs:
            st.subheader("🛠️ Quick Actions")
            col_export, col_clear = st.columns(2)
            
            with col_export:
                if st.button("📄 Export Report"):
                    report = {
                        "patient_age": age,
                        "drugs": [drug.__dict__ for drug in drugs],
                        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.download_button(
                        "📥 Download JSON Report",
                        data=json.dumps(report, indent=2),
                        file_name="prescription_analysis.json",
                        mime="application/json"
                    )
            
            with col_clear:
                if st.button("🗑️ Clear All"):
                    st.session_state.drug_count = 1
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7F8C8D; font-size: 0.9em;'>
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    Always consult healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()