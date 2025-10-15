import lzma
import base64
from typing import Dict

def encode_text_local(text: str, title: str) -> Dict[str, str]:
    """
    Lossless, deterministic encoding: lzma compress -> base64 encode.
    Returns a dict with title and encoded_data.
    Tunable preset=6 balances speed/ratio; crank to 9 for max compression.
    """
    compressed = lzma.compress(text.encode("utf-8"), preset=6)
    b64 = base64.b64encode(compressed).decode("ascii")
    return {"title": title, "encoded_data": b64}

def decode_text_local(encoded_data: str) -> str:
    """
    Decode base64 -> lzma decompress -> original string.
    """
    try:
        compressed = base64.b64decode(encoded_data.encode("ascii"))
        text = lzma.decompress(compressed).decode("utf-8")
        return text
    except Exception as e:
        raise ValueError(f"Decompression failed: {e}")