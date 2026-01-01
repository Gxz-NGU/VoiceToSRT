import os
import difflib
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import whisper

try:
    import stable_whisper
except Exception:
    stable_whisper = None

try:
    import torch
except Exception:
    torch = None

class VoiceAligner:
    def __init__(self, model_name="base"):
        """
        Initialize the aligner with a specific Whisper model.
        Args:
            model_name (str): The name of the Whisper model to use.
        """
        if torch is not None:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.model_name = model_name
        self.use_alignment = stable_whisper is not None
        if self.use_alignment:
            print(f"Loading Stable Whisper model: {model_name}...")
            self.model = stable_whisper.load_model(model_name)
        else:
            print(f"Loading Whisper model: {model_name}...")
            self.model = whisper.load_model(model_name)
        print("Model loaded.")

    def align_transcript(self, audio_path, text_path, output_srt_path, language=None):
        """
        Aligns a transcript text file to an audio file using segment-level interpolation.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found: {text_path}")

        # 1. Read user text
        with open(text_path, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]

        if not raw_lines:
            raise ValueError("Transcript file is empty.")

        if self.use_alignment:
            try:
                srt_segments = self._align_with_stable_whisper(
                    audio_path=audio_path,
                    raw_lines=raw_lines,
                    language=language
                )
                self._write_srt(srt_segments, output_srt_path)
                print("SRT generation complete.")
                return
            except Exception as e:
                print(f"Stable alignment failed: {e}. Falling back to transcription alignment.")

        srt_segments = self._align_with_transcription(
            audio_path=audio_path,
            raw_lines=raw_lines,
            language=language
        )
        self._write_srt(srt_segments, output_srt_path)
        print("SRT generation complete.")

    def _align_with_stable_whisper(self, audio_path, raw_lines, language=None):
        if stable_whisper is None:
            raise RuntimeError("stable_whisper is not available.")

        if language is None:
            language = self._detect_language(audio_path)
            print(f"Detected language: {language}")

        print("Aligning transcript with audio (stable_whisper)...")
        text = "\n".join(raw_lines)
        result = self.model.align(
            audio_path,
            text,
            language=language,
            original_split=True
        )

        if result is None or not result.segments:
            raise RuntimeError("Alignment returned no segments.")
        if len(result.segments) != len(raw_lines):
            raise RuntimeError(
                f"Aligned segments ({len(result.segments)}) do not match transcript lines ({len(raw_lines)})."
            )

        srt_segments = []
        for idx, seg in enumerate(result.segments):
            start = seg.start if seg.start is not None else 0.0
            end = seg.end if seg.end is not None else start
            if end < start:
                end = start
            srt_segments.append({
                "index": idx + 1,
                "start": start,
                "end": end,
                "text": raw_lines[idx]
            })

        return srt_segments

    def _align_with_transcription(self, audio_path, raw_lines, language=None):
        print(f"Transcribing {audio_path} for fallback alignment...")
        result = self.model.transcribe(
            audio_path,
            word_timestamps=False,
            language=language
        )

        if hasattr(result, "segments"):
            segments = result.segments
        else:
            segments = result.get("segments") or []
        if not segments:
            raise RuntimeError("No transcription segments returned.")
        print(f"Got {len(segments)} segments from audio.")

        full_transcript_text = ""
        char_timestamps = []
        for seg in segments:
            seg_text = self._normalize(seg.text if hasattr(seg, "text") else seg["text"])
            if not seg_text:
                continue

            start = seg.start if hasattr(seg, "start") else seg["start"]
            end = seg.end if hasattr(seg, "end") else seg["end"]
            duration = end - start
            char_count = len(seg_text)

            if char_count > 0 and duration > 0:
                time_per_char = duration / char_count
                for i in range(char_count):
                    t = start + (i * time_per_char)
                    char_timestamps.append(t)
                full_transcript_text += seg_text

        if not full_transcript_text or not char_timestamps:
            raise RuntimeError("Failed to build character map from transcription.")

        print(f"Built character map. Total chars: {len(full_transcript_text)}")
        last_end = segments[-1].end if hasattr(segments[-1], "end") else segments[-1]["end"]
        avg_time_per_char = (last_end / len(full_transcript_text))

        srt_segments = []
        global_cursor = 0
        last_end_time = 0.0
        normalized_lines = [self._normalize(line) for line in raw_lines]

        for line_idx, (raw_line, norm_line) in enumerate(zip(raw_lines, normalized_lines)):
            if not norm_line:
                continue

            search_start = global_cursor
            search_end = min(len(full_transcript_text), global_cursor + len(norm_line) * 3 + 100)
            search_block = full_transcript_text[search_start:search_end]

            matcher = difflib.SequenceMatcher(None, norm_line, search_block)
            match = matcher.find_longest_match(0, len(norm_line), 0, len(search_block))

            if match.size > 0:
                abs_start_idx = search_start + match.b
            else:
                print(f"Warning: Could not confidently match line: {raw_line[:20]}...")
                abs_start_idx = min(global_cursor, len(full_transcript_text) - 1)

            abs_end_idx = abs_start_idx + len(norm_line)

            t_start = self._time_for_char_idx(char_timestamps, abs_start_idx, avg_time_per_char)
            if t_start < last_end_time:
                t_start = last_end_time
            t_end = self._time_for_char_idx(char_timestamps, abs_end_idx, avg_time_per_char)
            if t_end < t_start:
                t_end = t_start

            global_cursor = min(abs_end_idx, len(full_transcript_text))
            last_end_time = t_end

            srt_segments.append({
                "index": line_idx + 1,
                "start": t_start,
                "end": t_end,
                "text": raw_line
            })

        return srt_segments

    def _detect_language(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

    def _time_for_char_idx(self, char_timestamps, index, avg_time_per_char):
        if not char_timestamps:
            return 0.0
        if index < len(char_timestamps):
            return char_timestamps[index]
        extra = index - (len(char_timestamps) - 1)
        return char_timestamps[-1] + (extra * avg_time_per_char)

    def _normalize(self, text):
        # Remove punctuation and spaces
        return re.sub(r'[^\w]', '', text).lower()

    def _format_timestamp(self, seconds):
        # Convert seconds to SRT timestamp format: HH:MM:SS,mmm
        millis = int((seconds % 1) * 1000)
        seconds = int(seconds)
        minutes = seconds // 60
        hours = minutes // 60
        minutes = minutes % 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _write_srt(self, segments, path):
        with open(path, 'w', encoding='utf-8') as f:
            for seg in segments:
                start = self._format_timestamp(seg['start'])
                end = self._format_timestamp(seg['end'])
                f.write(f"{seg['index']}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg['text']}\n\n")

if __name__ == "__main__":
    pass
