"""Write tests/golden/expected/ from the current code (see tests/golden/README.md).

Run on the reference machine before a change that could move numbers, and again only after a change that
is meant to (marked ▲ in the changelog).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.golden.outputs import write_expected  # noqa: E402

if __name__ == '__main__':
    if sys.argv[1:] == ['--family', 'continuous']:
        from tests.golden.continuous.outputs import write_expected as write_continuous  # noqa: E402
        write_continuous()
    else:
        write_expected()
