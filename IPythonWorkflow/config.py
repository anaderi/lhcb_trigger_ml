# Global constants,
# if you need to override them, do it in
# config.local.py (create the file if it doesn't exist)
ipc_cluster = None

try:
    import config.local
except:
    pass
