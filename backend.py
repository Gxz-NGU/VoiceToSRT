import whisper
import os
import difflib
import re

class VoiceAligner:
    def __init__(self, model_name="base"):
        """
        Initialize the aligner with a specific Whisper model.
        Args:
            model_name (str): The name of the Whisper model to use.
        """
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

        # 2. Transcribe audio (Segment level only)
        print(f"Transcribing {audio_path}...")
        result = self.model.transcribe(
            audio_path, 
            word_timestamps=False, # Avoid segfault
            language=language or "ko" # Default to ko or detect
        )
        
        segments = result["segments"]
        print(f"Got {len(segments)} segments from audio.")

        # 3. Build Character-Time Map
        # We create a long normalized string of the transcription and a corresponding list of timestamps for each char.
        full_transcript_text = ""
        char_timestamps = [] # List of time values
        
        for seg in segments:
            seg_text = self._normalize(seg["text"])
            if not seg_text:
                continue
                
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            char_count = len(seg_text)
            
            # Interpolate
            if char_count > 0:
                time_per_char = duration / char_count
                for i in range(char_count):
                    t = start + (i * time_per_char)
                    char_timestamps.append(t)
                
                full_transcript_text += seg_text
            
        print(f"Built character map. Total chars: {len(full_transcript_text)}")

        # 4. Align user lines to the full transcript
        srt_segments = []
        global_cursor = 0
        
        # We will search for each user line in the full transcript text
        # starting from where the last one ended (roughly).
        
        normalized_lines = [self._normalize(line) for line in raw_lines]
        
        for line_idx, (raw_line, norm_line) in enumerate(zip(raw_lines, normalized_lines)):
            if not norm_line:
                continue
                
            # Search for this line in the transcript
            # We use SequenceMatcher to find the best match block starting near global_cursor
            
            # Define search window: e.g. next 500 chars from cursor
            # (assuming user text roughly matches transcript)
            
            search_start = global_cursor
            search_end = min(len(full_transcript_text), global_cursor + len(norm_line) * 3 + 100)
            search_block = full_transcript_text[search_start:search_end]
            
            matcher = difflib.SequenceMatcher(None, norm_line, search_block)
            match = matcher.find_longest_match(0, len(norm_line), 0, len(search_block))
            
            if match.size > 0:
                # Calculate absolute indices in full_transcript_text
                abs_start_idx = search_start + match.b
                abs_end_idx = abs_start_idx + match.size
                
                # Check bounds
                if abs_start_idx < len(char_timestamps):
                    t_start = char_timestamps[abs_start_idx]
                else:
                    t_start = char_timestamps[-1]
                    
                if abs_end_idx < len(char_timestamps):
                    t_end = char_timestamps[abs_end_idx]
                else:
                    # Look ahead slightly?
                    if abs_end_idx >= len(char_timestamps):
                        t_end = char_timestamps[-1] + 1.0 # estimation
                    else:
                        t_end = char_timestamps[abs_end_idx]
                
                # Update global cursor to end of this match
                # But allow some overlap/error correction? No, advance.
                global_cursor = abs_end_idx
                
                srt_segments.append({
                    "index": line_idx + 1,
                    "start": t_start,
                    "end": t_end,
                    "text": raw_line
                })
            else:
                # No match found? 
                print(f"Warning: Could not match line: {raw_line[:20]}...")
                # Best effort: Assume it takes up 2 seconds after the last one?
                # Or skip.
                pass

        # 5. Generate SRT
        self._write_srt(srt_segments, output_srt_path)
        print("SRT generation complete.")

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
