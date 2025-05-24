# art-restoration-ai
AI model that visually restores deteriorated paintings to guide and support art conservators during treatment.

## Setup

### Install the environment
From the project root, run:
`bash setup.sh`

This will:

- Create the art-restoration-ai Conda environment

- Install dependencies from configs/environment.yml

- Register the environment as a Jupyter kernel

### Importing Damage Effects
Add this to the top of your notebook to enable imports from src/:
```
import sys, os

project_root = os.path.abspath("..")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
```

Then you can import:
```
from damage_effects import TearDamage
```
