"""Streamlit entry script shipped inside the package.

``root-explorer`` and ``python -m root_explorer`` run this file with Streamlit, so
installed wheels do not depend on the repository-root ``streamlit_app.py``.
"""

from root_explorer.app.main import main

if __name__ == "__main__":
    main()
