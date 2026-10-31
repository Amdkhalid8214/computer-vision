# PowerShell launcher for HANDVISION STUDIO — REALITYFRAME EDITION
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\.venv\Scripts\python.exe" "$ScriptDir\main.py" @args

