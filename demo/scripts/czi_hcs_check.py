"""Launch the installed CZI HCS inspector from a source checkout."""

import sys

from czitools.hcs_check import main

if __name__ == "__main__":
    sys.exit(main())
