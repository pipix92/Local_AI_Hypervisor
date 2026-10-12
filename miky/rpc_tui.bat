@echo off
chcp 65001 >nul
rem Avvia la TUI del ggml-rpc-server di Miky
python "C:\Users\mikye\hermes-rpc\rpc_tui.py" %*
pause
