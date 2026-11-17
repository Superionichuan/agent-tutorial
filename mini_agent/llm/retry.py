"""Trying a model call again after a passing failure.

A rate limit (HTTP 429), a server error (HTTP 5xx) or a timeout is tried again,
up to 3 more times, after waiting 1, 2 and then 4 seconds; a Retry-After header
from the server makes a wait longer (at most 30 seconds). Any other failure
(400 bad request, 401 bad key, 403 not allowed, connection refused, ...) is
reported at once: trying again would not change it.

A backend sends through send(), with its own retry_waits (tests set zeros).
"""
import time
from typing import Callable, Optional, Sequence

import httpx

WAITS = (1.0, 2.0, 4.0)       # seconds before the first, second and third retry
MAX_WAIT = 30.0


def retryable(status: int) -> bool:
    """A status worth trying again: 429 (too many requests) or 5xx (server error)."""
    return status == 429 or 500 <= status <= 599


def wait_before(retry: int, response: Optional[httpx.Response] = None,
                waits: Sequence[float] = WAITS) -> float:
    """Seconds to wait before retry number `retry` (0 = the first retry)."""
    wait = float(waits[min(retry, len(waits) - 1)]) if waits else 0.0
    if response is not None:
        try:
            wait = max(wait, float(response.headers.get("retry-after", "")))
        except ValueError:                    # absent, or an HTTP date: use our own wait
            pass
    return max(0.0, min(wait, MAX_WAIT))


def complete(response: httpx.Response) -> httpx.Response:
    """Read the body of a response requested with stream=True. When reading the body of a
    final error (400, 401, 403, ...) times out, that error is returned without its message,
    so it is not tried again; any other timeout is raised (and tried again, as a timeout)."""
    try:
        response.read()
        return response
    except httpx.TimeoutException:
        response.close()
        if response.status_code >= 400 and not retryable(response.status_code):
            return httpx.Response(response.status_code, request=response.request,
                                  content=b"(timed out while reading the error message)")
        raise


def send(request: Callable[[], httpx.Response], waits: Sequence[float] = WAITS,
         sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    """request() again after a retryable status or a timeout, at most len(waits) times.
    Returns the last response (possibly still an error), or raises the last timeout."""
    for retry in range(len(waits) + 1):
        last = retry == len(waits)
        try:
            response = request()
        except httpx.TimeoutException:
            if last:
                raise
            sleep(wait_before(retry, None, waits))
            continue
        if last or not retryable(response.status_code):
            return response
        sleep(wait_before(retry, response, waits))
    raise AssertionError("unreachable")
