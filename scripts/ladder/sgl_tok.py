#!/usr/bin/env python3
"""sgl_sim.py on a token stream that is ALREADY what the server saw (make_token_stream.py output): skip sgl_sim's
replay-client emulation (which would replace every output with program-unique ids) and replay the ids verbatim.
All sgl_sim.py arguments pass through; do not give --ids-from.   (sgl_sim.py itself is untouched.)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
import sgl_sim   # patches the clock on import, as when run directly

sgl_sim.emulate_client = lambda reqs, **kw: reqs
if __name__ == "__main__":
    sgl_sim.main()
