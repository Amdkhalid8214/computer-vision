# PowerShell test runner for HANDVISION STUDIO — REALITYFRAME EDITION
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\.venv\Scripts\python.exe" -m pytest @args

