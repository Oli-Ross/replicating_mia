"""
.. include:: ../docs/utils.md
"""

from typing import Dict
import hashlib

def hash(string:str) -> str:
    h = hashlib.new('sha256')
    h.update(string.encode())
    return h.hexdigest()
