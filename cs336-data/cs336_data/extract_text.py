import resiliparse.parse
import resiliparse.extract.html2text

def extract_text(html_bytes: bytes):
    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    # Add error handling to the decode operation
    html_str = html_bytes.decode(encoding, errors='replace')
    return resiliparse.extract.html2text.extract_plain_text(html_str)
