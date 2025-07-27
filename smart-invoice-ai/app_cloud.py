import streamlit as st

st.title("🤖 Smart Invoice AI System")
st.write("Hello World! The system is working.")

if st.button("Test Button"):
    st.success("✅ Button works!")
    st.balloons()

st.markdown("---")
st.markdown("### 🎯 Project Features")
st.write("- AI-powered invoice processing")
st.write("- Real-time learning from corrections")
st.write("- Multi-format support (PDF, images, CSV)")
st.write("- Professional web interface")

st.info("🔧 Full functionality coming soon!")
# Test OCR imports
try:
    import pytesseract
    import pdf2image

    st.success("✅ OCR modules imported successfully!")

    # Test Tesseract version
    version = pytesseract.get_tesseract_version()
    st.info(f"📋 Tesseract version: {version}")
except Exception as e:
    st.error(f"❌ OCR import error: {e}")