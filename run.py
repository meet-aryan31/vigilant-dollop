import argparse

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a file and send it to an LLM."
    )
    parser.add_argument("file", help="Path to audio/image/PDF/TXT/DOCX file")
    args = parser.parse_args()

    response = run_pipeline(args.file)
    print(response if response is not None else "null")


if __name__ == "__main__":
    main()
