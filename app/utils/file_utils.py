import os


def save_bytes_to_file(path: str, data: bytes) -> str:
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path

