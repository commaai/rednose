import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
HELPERS_PATH = os.path.join(BASEDIR, "helpers")
TEMPLATE_DIR = os.path.join(BASEDIR, "templates")
# For consumers that need the parent dir in include paths
INCLUDE_PATH = os.path.abspath(os.path.join(BASEDIR, "../"))
