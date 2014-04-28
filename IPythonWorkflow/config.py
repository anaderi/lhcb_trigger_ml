# Global constants,
# if you need to override them, do it in
# config.local.py (create the file if it doesn't exist)
ipc_cluster = None

try:
    import configlocal
except ImportError:
    print("No configlocal.py file was found, using default config")
