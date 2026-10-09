import tempfile
import unittest
from pathlib import Path
from support import Scripted, reply, run
from mini_agent.llm.offline import RecordingLLM


class Regression(unittest.TestCase):
    def test_nonstreaming_recording_wrapper_still_emits_content(self):
        response = reply("done", {"answer": "visible answer"})
        with tempfile.TemporaryDirectory() as directory:
            wrapped = RecordingLLM(Scripted(response), Path(directory) / "recording.jsonl")
            chunks = []
            actual = wrapped.call_stream("question", lambda text, **kw: chunks.append(text))
        self.assertEqual(actual, response)
        self.assertEqual("".join(chunks), response,
                         f"stream callback received no answer; chunks={chunks!r}")


if __name__ == "__main__":
    unittest.main()
