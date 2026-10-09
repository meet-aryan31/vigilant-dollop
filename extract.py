import argparse

from text_extract import extract_text


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from an image, PDF, TXT, or DOCX file."
    )
    parser.add_argument("file", help="Path to the file")
    args = parser.parse_args()

    text = extract_text(args.file)
    print(text if text is not None else "null")


if __name__ == "__main__":
    main()
