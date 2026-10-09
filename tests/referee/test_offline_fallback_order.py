import tempfile
import unittest
from pathlib import Path
from support import recording
from mini_agent.llm.offline import OfflineLLM


class Regression(unittest.TestCase):
    def test_fallback_replays_repeated_exchanges_in_order(self):
        OfflineLLM._cursor.clear()
        prompt = "[Goal]\nsame goal\n\n[Tools]\nold catalog\n\n[Steps taken so far]\n(none)"
        changed = prompt.replace("old catalog", "new catalog")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.jsonl"
            recording(path, prompt, "first", "second")
            llm = OfflineLLM(path)
            actual = [llm.call(changed), llm.call(changed)]
            self.assertEqual(actual, ["first", "second"], f"fallback replay sequence={actual!r}")


if __name__ == "__main__":
    unittest.main()
