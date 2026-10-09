import tempfile
import unittest
from pathlib import Path
from support import Scripted
from mini_agent.llm.offline import OfflineLLM, RecordingLLM, NO_RECORDING, describe


class Regression(unittest.TestCase):
    def test_meaningful_date_change_must_not_replay_old_answer(self):
        OfflineLLM._cursor.clear()
        old = "Return only the year component of 2024-01-01."
        new = "Return only the year component of 2025-01-01."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.jsonl"
            RecordingLLM(Scripted("2024"), path).call(old)
            actual = OfflineLLM(path).call(new)
        self.assertEqual(actual, NO_RECORDING,
                         f"date-sensitive new question replayed {actual!r}; equal keys={describe(old)['key'] == describe(new)['key']}")


if __name__ == "__main__":
    unittest.main()
