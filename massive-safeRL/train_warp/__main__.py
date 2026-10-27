from __future__ import annotations

import sys
from massive_saferl.__main__ import main


if __name__ == '__main__':
    main(['train_warp', *sys.argv[1:]])
