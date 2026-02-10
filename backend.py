import os
import difflib
import re
from typing import List, Dict, Tuple

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
        self.min_duration_for_short = 0.12
        self.max_zero_ratio = 0.12
        self.max_short_ratio = 0.35
        self.max_large_gap = 8.0
        self.max_trailing_stuck = 3
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
            stable_errors = []

            try:
                srt_segments = self._align_with_stable_whisper(
                    audio_path=audio_path,
                    raw_lines=raw_lines,
                    language=language
                )
                quality = self._evaluate_alignment_quality(srt_segments)
                print(f"Primary stable alignment quality: {self._format_quality(quality)}")
                if quality["ok"]:
                    self._write_srt(srt_segments, output_srt_path)
                    print("SRT generation complete.")
                    return
                stable_errors.append(f"primary quality low: {self._format_quality(quality)}")
            except Exception as e:
                stable_errors.append(f"primary failed: {e}")

            try:
                srt_segments = self._align_with_grouped_stable_whisper(
                    audio_path=audio_path,
                    raw_lines=raw_lines,
                    language=language
                )
                quality = self._evaluate_alignment_quality(srt_segments)
                print(f"Grouped stable alignment quality: {self._format_quality(quality)}")
                if quality["ok"]:
                    self._write_srt(srt_segments, output_srt_path)
                    print("SRT generation complete.")
                    return
                stable_errors.append(f"grouped quality low: {self._format_quality(quality)}")
            except Exception as e:
                stable_errors.append(f"grouped failed: {e}")

            print(
                "Stable alignment failed quality checks. Falling back to transcription alignment. "
                f"Reasons: {' | '.join(stable_errors)}"
            )

        srt_segments = self._align_with_transcription(
            audio_path=audio_path,
            raw_lines=raw_lines,
            language=language
        )
        fallback_quality = self._evaluate_alignment_quality(srt_segments)
        print(f"Fallback alignment quality: {self._format_quality(fallback_quality)}")
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
            original_split=True,
            fast_mode=True
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

    def _align_with_grouped_stable_whisper(self, audio_path, raw_lines, language=None):
        if stable_whisper is None:
            raise RuntimeError("stable_whisper is not available.")

        if language is None:
            language = self._detect_language(audio_path)
            print(f"Detected language: {language}")

        groups = self._build_alignment_groups(raw_lines)
        grouped_text = "\n".join(group["text"] for group in groups)

        print(f"Aligning transcript with grouped strategy (groups={len(groups)})...")
        result = self.model.align(
            audio_path,
            grouped_text,
            language=language,
            original_split=True,
            fast_mode=True
        )

        if result is None or not result.segments:
            raise RuntimeError("Grouped alignment returned no segments.")
        if len(result.segments) != len(groups):
            raise RuntimeError(
                f"Grouped aligned segments ({len(result.segments)}) do not match group count ({len(groups)})."
            )

        srt_segments = []
        for group_idx, (seg, group) in enumerate(zip(result.segments, groups)):
            start = seg.start if seg.start is not None else 0.0
            end = seg.end if seg.end is not None else start
            if end < start:
                end = start

            spans = self._split_span_by_weights(start, end, group["weights"])
            for line_no, (line_index, raw_line) in enumerate(group["lines"]):
                line_start, line_end = spans[line_no]
                if line_end < line_start:
                    line_end = line_start
                srt_segments.append({
                    "index": line_index + 1,
                    "start": line_start,
                    "end": line_end,
                    "text": raw_line
                })

        if len(srt_segments) != len(raw_lines):
            raise RuntimeError(
                f"Grouped expansion produced {len(srt_segments)} segments for {len(raw_lines)} lines."
            )

        return srt_segments

    def _build_alignment_groups(self, raw_lines: List[str]) -> List[Dict]:
        groups = []
        current_lines: List[Tuple[int, str]] = []
        current_chars = 0

        min_chars = 40
        max_lines = 5

        for idx, line in enumerate(raw_lines):
            norm_len = max(len(self._normalize(line)), 1)
            current_lines.append((idx, line))
            current_chars += norm_len

            if current_chars >= min_chars or len(current_lines) >= max_lines:
                weights = [max(len(self._normalize(text)), 1) for _, text in current_lines]
                groups.append({
                    "lines": current_lines,
                    "weights": weights,
                    "text": " ".join(text for _, text in current_lines)
                })
                current_lines = []
                current_chars = 0

        if current_lines:
            weights = [max(len(self._normalize(text)), 1) for _, text in current_lines]
            groups.append({
                "lines": current_lines,
                "weights": weights,
                "text": " ".join(text for _, text in current_lines)
            })

        return groups

    def _split_span_by_weights(self, start: float, end: float, weights: List[int]) -> List[Tuple[float, float]]:
        if not weights:
            return []

        if len(weights) == 1:
            return [(start, max(start, end))]

        duration = max(0.0, end - start)
        if duration == 0:
            return [(start, start) for _ in weights]

        total = float(sum(max(w, 1) for w in weights))
        spans = []
        cursor = start
        for i, w in enumerate(weights):
            if i == len(weights) - 1:
                next_t = end
            else:
                next_t = cursor + (duration * (max(w, 1) / total))
            if next_t < cursor:
                next_t = cursor
            spans.append((cursor, next_t))
            cursor = next_t
        return spans

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

        asr_text = ""
        asr_char_timestamps = []
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
                    asr_char_timestamps.append(t)
                asr_text += seg_text

        if not asr_text or not asr_char_timestamps:
            raise RuntimeError("Failed to build character map from transcription.")

        print(f"Built character map. Total ASR chars: {len(asr_text)}")
        last_end = segments[-1].end if hasattr(segments[-1], "end") else segments[-1]["end"]
        avg_time_per_char = (last_end / len(asr_text))

        ref_text = ""
        line_ranges = []
        for raw_line in raw_lines:
            norm_line = self._normalize(raw_line)
            start_idx = len(ref_text)
            ref_text += norm_line
            end_idx = len(ref_text)
            line_ranges.append((start_idx, end_idx))

        if not ref_text:
            raise RuntimeError("Transcript lines became empty after normalization.")

        ref_to_asr, match_ratio = self._build_ref_to_asr_map(ref_text, asr_text)
        print(f"Fallback global char alignment match ratio: {match_ratio:.3f}")

        srt_segments = []
        last_end_time = 0.0
        for line_idx, raw_line in enumerate(raw_lines):
            ref_start, ref_end = line_ranges[line_idx]

            if ref_start >= len(ref_to_asr):
                asr_start_idx = len(asr_char_timestamps) - 1
            else:
                asr_start_idx = ref_to_asr[ref_start]

            if ref_end <= ref_start:
                asr_end_idx = asr_start_idx
            else:
                mapped_end_pos = min(ref_end - 1, len(ref_to_asr) - 1)
                asr_end_idx = ref_to_asr[mapped_end_pos] + 1

            t_start = self._time_for_char_idx(asr_char_timestamps, asr_start_idx, avg_time_per_char)
            if t_start < last_end_time:
                t_start = last_end_time
            t_end = self._time_for_char_idx(asr_char_timestamps, asr_end_idx, avg_time_per_char)
            if t_end < t_start:
                t_end = t_start

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

    def _build_ref_to_asr_map(self, ref_text: str, asr_text: str) -> Tuple[List[int], float]:
        if not ref_text or not asr_text:
            raise RuntimeError("Reference text and ASR text must be non-empty.")

        raw_map = [-1] * len(ref_text)
        matcher = difflib.SequenceMatcher(None, ref_text, asr_text, autojunk=False)
        matched_chars = 0
        for block in matcher.get_matching_blocks():
            if block.size <= 0:
                continue
            matched_chars += block.size
            for offset in range(block.size):
                raw_map[block.a + offset] = block.b + offset

        left_ref = [-1] * len(ref_text)
        left_asr = [-1] * len(ref_text)
        last_ref = -1
        last_asr = -1
        for i, asr_idx in enumerate(raw_map):
            if asr_idx != -1:
                last_ref = i
                last_asr = asr_idx
            left_ref[i] = last_ref
            left_asr[i] = last_asr

        right_ref = [-1] * len(ref_text)
        right_asr = [-1] * len(ref_text)
        next_ref = -1
        next_asr = -1
        for i in range(len(ref_text) - 1, -1, -1):
            asr_idx = raw_map[i]
            if asr_idx != -1:
                next_ref = i
                next_asr = asr_idx
            right_ref[i] = next_ref
            right_asr[i] = next_asr

        max_asr_idx = len(asr_text) - 1
        mapped = []
        prev = 0
        for i, asr_idx in enumerate(raw_map):
            if asr_idx == -1:
                l_ref = left_ref[i]
                l_asr = left_asr[i]
                r_ref = right_ref[i]
                r_asr = right_asr[i]

                if l_ref != -1 and r_ref != -1 and r_ref != l_ref:
                    ratio = (i - l_ref) / (r_ref - l_ref)
                    est = l_asr + ratio * (r_asr - l_asr)
                    asr_idx = int(round(est))
                elif l_ref != -1:
                    asr_idx = l_asr + (i - l_ref)
                elif r_ref != -1:
                    asr_idx = r_asr - (r_ref - i)
                else:
                    asr_idx = 0

            if asr_idx < 0:
                asr_idx = 0
            if asr_idx > max_asr_idx:
                asr_idx = max_asr_idx
            if asr_idx < prev:
                asr_idx = prev

            mapped.append(asr_idx)
            prev = asr_idx

        match_ratio = matched_chars / len(ref_text)
        return mapped, match_ratio

    def _evaluate_alignment_quality(self, segments: List[Dict]) -> Dict:
        if not segments:
            return {
                "ok": False,
                "count": 0,
                "zero_count": 0,
                "short_count": 0,
                "max_gap": 0.0,
                "trailing_stuck": 0,
                "zero_ratio": 1.0,
                "short_ratio": 1.0
            }

        durations = [max(0.0, seg["end"] - seg["start"]) for seg in segments]
        zero_count = sum(1 for d in durations if d <= 1e-6)
        short_count = sum(1 for d in durations if d < self.min_duration_for_short)

        gaps = []
        for i in range(len(segments) - 1):
            gaps.append(max(0.0, segments[i + 1]["start"] - segments[i]["end"]))

        max_gap = max(gaps) if gaps else 0.0
        trailing_stuck = self._count_trailing_stuck(segments)

        count = len(segments)
        zero_ratio = zero_count / count
        short_ratio = short_count / count

        ok = (
            zero_ratio <= self.max_zero_ratio and
            short_ratio <= self.max_short_ratio and
            max_gap <= self.max_large_gap and
            trailing_stuck <= self.max_trailing_stuck
        )

        return {
            "ok": ok,
            "count": count,
            "zero_count": zero_count,
            "short_count": short_count,
            "max_gap": max_gap,
            "trailing_stuck": trailing_stuck,
            "zero_ratio": zero_ratio,
            "short_ratio": short_ratio
        }

    def _count_trailing_stuck(self, segments: List[Dict], tolerance: float = 1e-3) -> int:
        if not segments:
            return 0
        anchor = segments[-1]["end"]
        count = 0
        for seg in reversed(segments):
            if abs(seg["start"] - anchor) <= tolerance and abs(seg["end"] - anchor) <= tolerance:
                count += 1
            else:
                break
        return count

    def _format_quality(self, quality: Dict) -> str:
        return (
            f"ok={quality['ok']}, zero={quality['zero_count']}/{quality['count']}, "
            f"short={quality['short_count']}/{quality['count']}, "
            f"max_gap={quality['max_gap']:.3f}s, trailing_stuck={quality['trailing_stuck']}"
        )

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
