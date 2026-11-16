from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
if (PROJECT_PARENT / "KivuSmart").is_dir():
    if str(PROJECT_PARENT) not in sys.path:
        sys.path.insert(0, str(PROJECT_PARENT))
else:
    # Streamlit Cloud can use the KivuSmart directory itself as repository root.
    package = types.ModuleType("KivuSmart")
    package.__path__ = [str(PROJECT_ROOT)]
    sys.modules.setdefault("KivuSmart", package)

from KivuSmart.frontend.app import main


if __name__ == "__main__":
    main()
