"""
CanopyLens - Root Application Entrypoint
Delegates execution to backend/app.py for deployment compatibility on Streamlit Community Cloud and HuggingFace Spaces.
"""

import sys
from pathlib import Path

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.app import main

if __name__ == "__main__":
    main()
