import tempfile
import unittest
from pathlib import Path
from support import recording
from mini_agent.llm.offline import OfflineLLM


class Regression(unittest.TestCase):
    def test_reading_one_recording_does_not_advance_another(self):
        OfflineLLM._cursor.clear()
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            prompt = "Which response belongs to this recording?"
            recording(a, prompt, "A-first", "A-second")
            recording(b, prompt, "B-first", "B-second")
            self.assertEqual(OfflineLLM(a).call(prompt), "A-first")
            actual = OfflineLLM(b).call(prompt)
            self.assertEqual(actual, "B-first", f"fresh recording started at {actual!r}")


if __name__ == "__main__":
    unittest.main()
