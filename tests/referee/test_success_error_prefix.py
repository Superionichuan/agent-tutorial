import tempfile
import unittest
from pathlib import Path
from support import Scripted, reply, run
from mini_agent.tools.file.read import read_file


class Regression(unittest.TestCase):
    def test_successful_file_read_is_not_a_tool_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_text("Error budget report")
            llm = Scripted(reply("read_file", {"path": str(path)}),
                           reply("done", {"answer": "ok"}))
            _, events = run(llm, tools={"read_file": read_file})
        result, = [data for kind, data in events if kind == "tool/result"]
        self.assertEqual(result["content"], "Error budget report")
        self.assertFalse(result["is_error"], "successful file content was labelled is_error=True")


if __name__ == "__main__":
    unittest.main()
