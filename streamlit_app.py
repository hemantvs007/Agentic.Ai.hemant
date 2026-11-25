"""
streamlit_app.py - Entry point for Streamlit Community Cloud.

Streamlit re-runs the script on every interaction, so we execute main.py
with runpy (a plain `import main` would only run once and the UI would freeze).
"""

import os
import sys
import runpy

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)  # so main.py can import helper.py / orchestrator.py

runpy.run_path(os.path.join(APP_DIR, "main.py"), run_name="__main__")