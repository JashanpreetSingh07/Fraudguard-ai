"""
Streamlit Cloud Entry Point for FraudGuard AI
Redirects to dashboard/app.py for instant 1-click cloud deployment.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run dashboard
import dashboard.app
