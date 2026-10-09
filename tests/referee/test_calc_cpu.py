import subprocess
import sys
import unittest
from support import ROOT


class Regression(unittest.TestCase):
    def test_short_expression_is_rejected_before_exhausting_cpu(self):
        # Only the child receives resource limits. The wall timeout is a backstop.
        code = '''
import resource
from mini_agent.tools.math.calc import calc
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
print(calc("9**9**9"), flush=True)
'''
        try:
            child = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                   capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            self.fail("calculator exhausted the imposed wall-time budget")
        self.assertEqual(child.returncode, 0,
                         f"calculator child returncode={child.returncode}, stdout={child.stdout!r}")
        self.assertTrue(child.stdout.startswith("Error:"), child.stdout)


if __name__ == "__main__":
    unittest.main()
