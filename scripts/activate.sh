# Committed activation wrapper. Sources an optional, untracked local override
# (scripts/activate.local.sh) for machine-specific settings such as the
# corporate SSL_CERT_DIR workaround used with pip-system-certs.
_here="${BASH_SOURCE[0]:-$0}"
_dir="$(cd "$(dirname "$_here")" 2>/dev/null && pwd)"
if [ -n "$_dir" ] && [ -f "$_dir/activate.local.sh" ]; then
    . "$_dir/activate.local.sh"
fi
