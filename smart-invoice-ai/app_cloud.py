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
# Enhanced Tesseract discovery
try:
    import pytesseract
    import pdf2image
    import os
    import shutil
    import subprocess

    st.success("✅ OCR modules imported successfully!")

    # Try to find tesseract with multiple methods
    st.write("🔍 Searching for Tesseract...")

    # Method 1: Common paths
    possible_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/app/.apt/usr/bin/tesseract',
        '/opt/conda/bin/tesseract',
        '/home/adminuser/.local/bin/tesseract',
        shutil.which('tesseract')
    ]

    # Method 2: Use 'which' command
    try:
        result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
        if result.returncode == 0:
            possible_paths.append(result.stdout.strip())
    except:
        pass

    # Method 3: Search filesystem
    try:
        result = subprocess.run(['find', '/usr', '-name', 'tesseract', '-type', 'f'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for path in result.stdout.strip().split('\n'):
                if path:
                    possible_paths.append(path)
    except:
        pass

    st.write("Searching paths:", possible_paths)

    tesseract_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            tesseract_path = path
            break

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        st.success(f"✅ Tesseract found at: {tesseract_path}")

        version = pytesseract.get_tesseract_version()
        st.info(f"📋 Tesseract version: {version}")

    else:
        st.error("❌ Tesseract still not found")

        # Show what's available in common directories
        st.write("📁 Contents of /usr/bin (first 20 files):")
        try:
            files = os.listdir('/usr/bin')[:20]
            st.write(files)
        except:
            st.write("Cannot access /usr/bin")

except Exception as e:
    st.error(f"❌ OCR error: {e}")