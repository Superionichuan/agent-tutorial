import tempfile
import unittest
from pathlib import Path
from support import Scripted, ROOT
from mini_agent.llm.offline import RecordingLLM


class Regression(unittest.TestCase):
    def test_record_does_not_store_workspace_path_from_history(self):
        workspace = "/private/customer-project"
        prompt = ("[Goal]\nRead notes\n\n[Tools]\nWorkspace: " + workspace
                  + "\n\n[Steps taken so far]\n"
                  + "1. read_file({'path': '" + workspace + "/notes.txt'}) -> notes")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.jsonl"
            RecordingLLM(Scripted("ok"), path).call(prompt)
            raw = path.read_text()
        self.assertNotIn(workspace, raw, "recording persisted the workspace path in history")


if __name__ == "__main__":
    unittest.main()
