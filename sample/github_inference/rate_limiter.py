"""
Rate limiter module for GitHub API interactions.

Author: Ron Webb
Since: 1.0.0
"""

import time
from typing import Any, Optional
import requests

# Constants for rate limiting
LOG_PREFIX = "Rate Limiter:"
HEADER_RATE_LIMIT_REMAINING = "x-ratelimit-remaining"
HEADER_RATE_LIMIT_RESET = "x-ratelimit-reset"
HEADER_RATE_LIMIT_TIME_REMAINING = "x-ratelimit-timeremaining"
HEADER_RETRY_AFTER = "retry-after"
MSG_MAX_RETRIES_EXCEEDED = "Max retries exceeded"
MSG_MAX_RETRIES_EXCEEDED_ERROR = "Max retries exceeded, raising error"
MSG_MAX_RETRIES_EXCEEDED_EXCEPTION = "Max retries exceeded for exception, raising"
MSG_NO_WAIT_NEEDED = "No need to wait, rate limit has already reset"
MSG_REMAINING_REQUESTS = "Remaining requests:"
MSG_RESET_TIME = "Reset time:"
MSG_TIME_REMAINING = "Time remaining:"
MSG_SERVER_RETRY_AFTER = "Server requested retry after"
MSG_EXPONENTIAL_BACKOFF = "Using exponential backoff:"
MSG_SLEEPING_BEFORE_RETRY = "Sleeping for {} seconds before retry"
MSG_REQUEST_EXCEPTION = "Request exception occurred:"
MSG_EXCEPTION_BACKOFF = "Exception backoff:"


class RateLimiter:
    """
    Rate limiter implementation for GitHub API requests.

    Stateless implementation that avoids shared mutable state for thread safety.

    Author: Ron Webb
    Since: 1.0.0
    """

    def __should_wait_for_reset(self, reset_time: Optional[float]) -> bool:
        """
        Check if we should wait for rate limit reset.

        Args:
            reset_time: Unix timestamp when rate limit resets

        Returns:
            True if we should wait, False otherwise
        """
        if reset_time is None:
            return False

        current_time = time.time()
        should_wait = current_time < reset_time
        print(
            f"{LOG_PREFIX} Current time: {current_time}, "
            f"Reset time: {reset_time}, Should wait: {should_wait}"
        )
        return should_wait

    def __wait_for_reset(self, reset_time: float) -> None:
        """
        Wait until rate limit resets.

        Args:
            reset_time: Unix timestamp when rate limit resets
        """
        current_time = time.time()
        if current_time < reset_time:
            sleep_time = reset_time - current_time
            print(f"{LOG_PREFIX} Waiting {sleep_time:.2f} seconds for rate limit reset")
            time.sleep(sleep_time)
        else:
            print(f"{LOG_PREFIX} {MSG_NO_WAIT_NEEDED}")

    def __extract_rate_limit_info(
        self, response: requests.Response
    ) -> tuple[Optional[int], Optional[float]]:
        """
        Extract rate limit information from API response headers.

        Args:
            response: HTTP response from GitHub API

        Returns:
            Tuple of (remaining_requests, reset_time)
        """
        remaining_requests = None
        reset_time = None

        # GitHub Models API uses different header names
        if HEADER_RATE_LIMIT_REMAINING in response.headers:
            remaining_requests = int(response.headers[HEADER_RATE_LIMIT_REMAINING])
            print(f"{LOG_PREFIX} {MSG_REMAINING_REQUESTS} {remaining_requests}")

        if HEADER_RATE_LIMIT_RESET in response.headers:
            reset_time = float(response.headers[HEADER_RATE_LIMIT_RESET])
            print(f"{LOG_PREFIX} {MSG_RESET_TIME} {reset_time}")
        elif HEADER_RATE_LIMIT_TIME_REMAINING in response.headers:
            # GitHub Models API uses time remaining in seconds
            time_remaining = int(response.headers[HEADER_RATE_LIMIT_TIME_REMAINING])
            reset_time = time.time() + time_remaining
            print(
                f"{LOG_PREFIX} Time remaining: {time_remaining}s, Reset time: {reset_time}"
            )

        # print(f"{LOG_PREFIX} Available headers: {list(response.headers.keys())}")
        return remaining_requests, reset_time

    def __execute_http_request(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int
    ) -> requests.Response:
        """
        Execute the actual HTTP POST request.

        Args:
            url: The URL to make the request to
            headers: HTTP headers for the request
            payload: JSON payload for the request
            timeout: Request timeout in seconds

        Returns:
            HTTP response object
        """
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

    def __calculate_sleep_time(
        self, response: requests.Response, retry_count: int, base_delay: float
    ) -> int:
        """
        Calculate sleep time for rate limit handling.

        Args:
            response: HTTP response from the API
            retry_count: Current retry attempt number
            base_delay: Base delay for exponential backoff

        Returns:
            Sleep time in seconds
        """
        retry_after = response.headers.get(HEADER_RETRY_AFTER)
        if retry_after:
            sleep_time = int(retry_after)
            print(f"{LOG_PREFIX} {MSG_SERVER_RETRY_AFTER} {sleep_time} seconds")
        else:
            # Exponential backoff
            sleep_time = base_delay * (2**retry_count)
            print(f"{LOG_PREFIX} {MSG_EXPONENTIAL_BACKOFF} {sleep_time} seconds")
        return sleep_time

    def __handle_rate_limit_response(
        self, response: requests.Response, retry_count: int, max_retries: int
    ) -> int:
        """
        Handle rate limit response and calculate sleep time.

        Args:
            response: HTTP response from the API
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries

        Returns:
            Sleep time in seconds

        Raises:
            requests.exceptions.HTTPError: If max retries exceeded
        """
        sleep_time = self.__calculate_sleep_time(response, retry_count, 1.0)

        if retry_count >= max_retries:
            print(f"{LOG_PREFIX} {MSG_MAX_RETRIES_EXCEEDED_ERROR}")
            response.raise_for_status()  # Raise the final error

        print(f"{LOG_PREFIX} Sleeping for {sleep_time} seconds before retry")
        time.sleep(sleep_time)
        return sleep_time

    def __handle_request_exception(
        self,
        exc: requests.exceptions.RequestException,
        retry_count: int,
        max_retries: int,
    ) -> None:
        """
        Handle request exception with exponential backoff.

        Args:
            exc: The request exception that occurred
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries

        Raises:
            requests.exceptions.RequestException: If max retries exceeded
        """
        print(f"{LOG_PREFIX} {MSG_REQUEST_EXCEPTION} {exc}")
        if retry_count >= max_retries:
            print(f"{LOG_PREFIX} {MSG_MAX_RETRIES_EXCEEDED_EXCEPTION}")
            raise exc

        # Exponential backoff for other errors
        sleep_time = 1.0 * (2**retry_count)
        print(f"{LOG_PREFIX} {MSG_EXCEPTION_BACKOFF} {sleep_time} seconds")
        time.sleep(sleep_time)

    def __process_successful_response(
        self, response: requests.Response, reset_time: Optional[float]
    ) -> tuple[requests.Response, Optional[float]]:
        """
        Process a successful response and extract rate limit information.

        Args:
            response: HTTP response from the API
            reset_time: Current reset time

        Returns:
            Tuple of (response, updated_reset_time)

        Raises:
            requests.exceptions.HTTPError: If response indicates an error
        """
        _, new_reset_time = self.__extract_rate_limit_info(response)
        if new_reset_time:
            reset_time = new_reset_time

        # Raise for HTTP errors (except rate limits which are handled separately)
        response.raise_for_status()
        return response, reset_time

    def make_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int = 30,
        max_retries: int = 3,
    ) -> requests.Response:
        """
        Make a rate-limited HTTP request with exponential backoff.

        Uses stateless approach - rate limiting is handled based on API response headers only.

        Args:
            url: The URL to make the request to
            headers: HTTP headers for the request
            payload: JSON payload for the request
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on rate limit errors

        Returns:
            HTTP response object

        Raises:
            requests.exceptions.RequestException: If request fails after all retries
        """
        retry_count = 0
        reset_time: Optional[float] = None

        while retry_count <= max_retries:
            if self.__should_wait_for_reset(reset_time):
                self.__wait_for_reset(reset_time)  # type: ignore

            try:
                response = self.__execute_http_request(url, headers, payload, timeout)
                _, new_reset_time = self.__extract_rate_limit_info(response)

                if new_reset_time:
                    reset_time = new_reset_time

                if response.status_code in (403, 429):
                    self.__handle_rate_limit_response(
                        response, retry_count, max_retries
                    )
                    retry_count += 1
                    continue

                response, reset_time = self.__process_successful_response(
                    response, reset_time
                )
                return response

            except requests.exceptions.RequestException as exc:
                self.__handle_request_exception(exc, retry_count, max_retries)
                retry_count += 1

        # This should never be reached, but included for safety
        raise requests.exceptions.RequestException(MSG_MAX_RETRIES_EXCEEDED)
