import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# Load fine-tuned model
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("models//bert_model")
    tokenizer = BertTokenizer.from_pretrained("models//bert_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Simple mapping
label_map = {0: "Weak", 1: "Medium", 2: "Strong"}
color_map = {0: "#ff4444", 1: "#ffaa00", 2: "#00cc44"}

# Simple CSS
st.markdown("""
<style>
.stTextInput input {
    font-size: 18px;
    padding: 10px;
    border-radius: 5px;
}

.strength-bar {
    height: 30px;
    border-radius: 5px;
    color: white;
    text-align: center;
    line-height: 30px;
    font-weight: bold;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.simple-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #007bff;
}

.requirement {
    padding: 5px 0;
    font-size: 14px;
}

.met { color: green; }
.not-met { color: red; }
</style>
""", unsafe_allow_html=True)

# Simple header
st.title("🔐 Password Strength Checker")
st.write("Enter your password to check its strength in real-time")

# Password input
password = st.text_input("Password:", placeholder="Type your password here...")

# Real-time prediction function
def predict_strength(password):
    if not password:
        return 0, [1.0, 0.0, 0.0]
    
    inputs = tokenizer(password, return_tensors="pt", padding=True, truncation=True, max_length=16)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
    return predicted_class, probs[0]

# Password requirements check
def check_requirements(password):
    return {
        "At least 8 characters": len(password) >= 8,
        "Contains uppercase": bool(re.search(r'[A-Z]', password)),
        "Contains lowercase": bool(re.search(r'[a-z]', password)),
        "Contains numbers": bool(re.search(r'\d', password)),
        "Contains special characters": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }

# Main analysis
if password:
    # Get prediction
    predicted_class, probs = predict_strength(password)
    strength_label = label_map[predicted_class]
    confidence = probs[predicted_class].item() * 100
    
    # Strength bar
    st.markdown(f"""
    <div class="strength-bar" style="background-color: {color_map[predicted_class]};">
        {strength_label} ({confidence:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    
    # Simple metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Length", len(password))
    with col2:
        st.metric("Strength", strength_label)
    with col3:
        st.metric("Confidence", f"{confidence:.0f}%")
    
    # Requirements check
    with st.container():
        st.markdown('<div class="simple-card">', unsafe_allow_html=True)
        st.subheader("Password Requirements")
        
        requirements = check_requirements(password)
        for req, met in requirements.items():
            icon = "✅" if met else "❌"
            css_class = "met" if met else "not-met"
            st.markdown(f'<div class="requirement {css_class}">{icon} {req}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model confidence scores
    if st.expander("Show detailed confidence scores"):
        for i, label in label_map.items():
            score = probs[i].item() * 100
            st.write(f"**{label}**: {score:.2f}%")

else:
    # Empty state
    st.info("👆 Enter a password above to see the analysis")

# Simple tips
with st.expander("💡 Password Tips"):
    st.write("""
    - Use at least 12 characters
    - Mix uppercase and lowercase letters
    - Include numbers and special characters
    - Avoid common words and patterns
    - Don't reuse passwords across sites
    """)

# Footer
st.markdown("---")
st.caption("Powered by BERT AI model")