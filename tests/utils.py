import os
import pathlib

PATH = pathlib.Path(__file__).parent.resolve()

ASSETS_PATH = os.path.join(PATH, "assets")
ARTIFACTS_PATH = os.path.join(PATH, "artifacts")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_PATH, exist_ok=True)