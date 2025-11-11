"""
Script to verify the setup for screenshot functionality.
Run this to check if everything is configured correctly.
"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("Verifying Screenshot Setup")
print("=" * 60)

# Check 1: pdf2image
print("\n[1] Checking pdf2image...")
try:
    from pdf2image import convert_from_path
    print("✓ pdf2image is installed")
except ImportError:
    print("✗ pdf2image is NOT installed")
    print("  Install with: pip install pdf2image")
    sys.exit(1)

# Check 2: Pillow
print("\n[2] Checking Pillow...")
try:
    from PIL import Image
    print("✓ Pillow is installed")
except ImportError:
    print("✗ Pillow is NOT installed")
    print("  Install with: pip install Pillow")
    sys.exit(1)

# Check 3: Poppler path
print("\n[3] Checking Poppler path...")
poppler_paths = [
    r"F:\AI-Agents-Course-Ed\agents\1_foundations\career_conversations\poppler-25.07.0\Library\bin",
    os.path.join(os.path.dirname(__file__), "poppler-25.07.0", "Library", "bin"),
    os.path.join(os.getcwd(), "poppler-25.07.0", "Library", "bin"),
]

poppler_found = False
for path in poppler_paths:
    if path and os.path.exists(path):
        print(f"✓ Poppler found at: {path}")
        poppler_found = True
        break

if not poppler_found:
    print("✗ Poppler NOT found in expected locations")
    print("  Please ensure poppler is installed at one of:")
    for path in poppler_paths:
        print(f"    - {path}")

# Check 4: KB folder
print("\n[4] Checking KB folder...")
kb_path = os.path.join(os.path.dirname(__file__), "kb")
if os.path.exists(kb_path):
    pdf_files = list(Path(kb_path).rglob("*.pdf"))
    print(f"✓ KB folder exists with {len(pdf_files)} PDF files")
    if pdf_files:
        print(f"  Sample files:")
        for pdf in pdf_files[:3]:
            print(f"    - {pdf.name}")
else:
    print(f"✗ KB folder NOT found at: {kb_path}")

# Check 5: FAISS index
print("\n[5] Checking FAISS index...")
faiss_index = os.path.join(os.path.dirname(__file__), "models", "faiss", "index.faiss")
faiss_store = os.path.join(os.path.dirname(__file__), "models", "faiss", "store.jsonl")

if os.path.exists(faiss_index) and os.path.exists(faiss_store):
    print("✓ FAISS index exists")
    # Try to read it
    try:
        import faiss
        index = faiss.read_index(faiss_index)
        print(f"  Index loaded: {index.ntotal} vectors")
    except Exception as e:
        print(f"  ⚠ Could not read index: {e}")
else:
    print("⚠ FAISS index not found - will be built on first run")

# Check 6: Assignment images directory
print("\n[6] Checking assignment images directory...")
images_dir = os.path.join(os.path.dirname(__file__), "screenshots", "assignment_images")
os.makedirs(images_dir, exist_ok=True)
print(f"✓ Directory ready at: {images_dir}")

# Check 7: Test PDF conversion (if poppler found)
print("\n[7] Testing PDF to image conversion...")
if poppler_found and pdf_files:
    try:
        test_pdf = str(pdf_files[0])
        print(f"  Testing with: {os.path.basename(test_pdf)}")
        
        # Find poppler path
        poppler_path = None
        for path in poppler_paths:
            if path and os.path.exists(path):
                poppler_path = path
                break
        
        from pdf2image import convert_from_path
        images = convert_from_path(
            test_pdf,
            first_page=1,
            last_page=1,
            dpi=100,
            poppler_path=poppler_path
        )
        print(f"  ✓ Successfully converted 1 page to image")
        print(f"  Image size: {images[0].size}")
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ⚠ Skipping test (poppler or PDFs not found)")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
print("\nIf all checks passed, you should be able to:")
print("  1. Ask about assignments/projects")
print("  2. See debug messages in console")
print("  3. Receive screenshots with responses")
print("\nTo test, run your app and ask:")
print('  "Tell me in detail about a project or assignment you worked on"')


