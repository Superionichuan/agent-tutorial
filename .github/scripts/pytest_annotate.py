"""Run pytest and report each failing test as a GitHub error annotation.

Annotations appear on the run's page and can be read without signing in, which a job log
cannot. GitHub shows at most 10 error annotations per step: the first 9 failures get one
each, with the start of the traceback, and the 10th lists every failed test.

Usage: python .github/scripts/pytest_annotate.py [pytest arguments]
"""
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def escape(text):
    """A message after `::`."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text):
    """A value such as title=...: `:` and `,` would end it early."""
    return escape(text).replace(":", "%3A").replace(",", "%2C")


def main():
    report = os.path.join(tempfile.mkdtemp(), "report.xml")
    cmd = [sys.executable, "-m", "pytest", f"--junitxml={report}", "-o", "junit_family=xunit1",
           *sys.argv[1:]]
    code = subprocess.call(cmd)
    if code == 0 or not os.path.exists(report):
        return code
    failures = []
    for case in ET.parse(report).getroot().iter("testcase"):
        for problem in list(case.findall("failure")) + list(case.findall("error")):
            test_id = f"{case.get('file', '?')}::{case.get('name', '?')}"
            message = (problem.get("message") or "").strip()
            details = (problem.text or "").strip().splitlines()
            failures.append((case.get("file"), case.get("line"), test_id, message, details))
    for path, line, test_id, message, details in failures[:9]:
        location = (f"file={escape_property(path)},line={int(line) + 1},"
                    if path and line and line.isdigit() else "")
        body = message + "\n" + "\n".join(details[-25:])
        print(f"::error {location}title={escape_property(test_id)}::{escape(body[:6000])}")
    if failures:
        listing = "\n".join(f[2] for f in failures)
        print(f"::error title={len(failures)} failed test(s)::{escape(listing[:6000])}")
    return code


if __name__ == "__main__":
    sys.exit(main())
