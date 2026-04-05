"""WSGI entry point for SAM3 video Label Studio ML backend."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from label_studio_ml.api import init_app
from model import NewModel

app = init_app(model_class=NewModel)

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9090)),
    )
