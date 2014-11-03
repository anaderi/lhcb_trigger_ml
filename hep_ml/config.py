# Global constants,
# if you need to override them, do it in
# configlocal.py (create the file if it doesn't exist)

# If there is some machine-specific initialization code you want to run,
# you can place it in configlocal.py too
ipc_profile = None

try:
    from configlocal import *
except ImportError:
    print("No configlocal.py file was found, using default config")
