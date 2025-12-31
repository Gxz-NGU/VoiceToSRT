import argparse
import sys
from backend import VoiceAligner

def main():
    parser = argparse.ArgumentParser(description="VoiceToSRT: Align text transcript with audio to generate SRT.")
    parser.add_argument("--audio", required=True, help="Path to the audio file (e.g., .mp3, .wav)")
    parser.add_argument("--text", required=True, help="Path to the transcript text file")
    parser.add_argument("--output", required=True, help="Path to the output SRT file")
    parser.add_argument("--language", help="Language code (e.g., 'ko' for Korean, 'ja' for Japanese)")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")

    args = parser.parse_args()

    try:
        aligner = VoiceAligner(model_name=args.model)
        aligner.align_transcript(
            audio_path=args.audio,
            text_path=args.text,
            output_srt_path=args.output,
            language=args.language
        )
        print(f"Successfully generated {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
