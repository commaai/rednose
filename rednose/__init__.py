import os


INCLUDE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_PATH = os.path.join(os.path.dirname(__file__), "helpers", "libekf_sym.a")
SCONS_TOOL_PATH = os.path.join(os.path.dirname(__file__), "site_scons", "site_tools")
