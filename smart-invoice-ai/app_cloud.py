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
# Test OCR imports with PATH fixing
try:
    import pytesseract
    import pdf2image
    import os
    import shutil

    st.success("✅ OCR modules imported successfully!")

    # Try to find Tesseract in common cloud locations
    possible_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/app/.apt/usr/bin/tesseract',
        shutil.which('tesseract')
    ]

    tesseract_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            tesseract_path = path
            break

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        st.success(f"✅ Tesseract found at: {tesseract_path}")

        # Test version
        version = pytesseract.get_tesseract_version()
        st.info(f"📋 Tesseract version: {version}")

        # Test basic OCR
        st.info("🔍 Tesseract is ready for OCR processing!")

    else:
        st.error("❌ Tesseract not found in expected locations")
        st.write("Searched paths:", possible_paths)

except Exception as e:
    st.error(f"❌ OCR error: {e}")