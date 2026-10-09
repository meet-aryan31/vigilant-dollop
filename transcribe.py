import argparse

from stt import audio_to_text


def main():
    parser = argparse.ArgumentParser(description="Convert an audio file to text.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument(
        "-m",
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    args = parser.parse_args()

    text = audio_to_text(args.audio_file, args.model)
    print(text)


if __name__ == "__main__":
    main()
