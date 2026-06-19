import sys
from pathlib import Path

# Add the 'code' root directory to sys.path at position 0 to avoid importing ourselves
code_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(code_dir))

import main

if __name__ == "__main__":
    # Force the --run-evaluation flag if not already present
    if "--run-evaluation" not in sys.argv:
        sys.argv.append("--run-evaluation")
        
    main.main()
