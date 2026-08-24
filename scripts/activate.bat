@echo off
REM Committed activation wrapper. Sources an optional, untracked local override
REM (scripts/activate.local.bat) for machine-specific settings such as the
REM corporate SSL_CERT_DIR workaround used with pip-system-certs.
if exist "%~dp0activate.local.bat" call "%~dp0activate.local.bat"
