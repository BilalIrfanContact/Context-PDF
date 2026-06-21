def extract_text_from_markdown(data: bytes) -> str:
    if not data:
        return ""

    try:
        return data.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return ""
