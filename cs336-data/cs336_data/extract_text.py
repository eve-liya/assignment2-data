import resiliparse.parse
import resiliparse.extract.html2text

def extract_text(html_bytes: bytes):
    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding)
    return resiliparse.extract.html2text.extract_plain_text(html_str)

def main():
    warc_path = "data/"
    with open(warc_path, "rb") as f:
        moby_bytes = f.read()
    moby_expected_path = FIXTURES_PATH / "moby_extracted.txt"
    with open(moby_expected_path) as f:
        moby_expected_text = f.read()
    assert moby_expected_text == run_extract_text_from_html_bytes(moby_bytes)

if __name__ == "__main__":
    main()