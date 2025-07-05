"""
Commons module containing shared utilities for GitHub API interactions.

Author: Ron Webb
Since: 1.0.0
"""

import time
from typing import Dict, Any, Optional
import requests


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
        print(f"Rate Limiter: Current time: {current_time}, Reset time: {reset_time}, Should wait: {should_wait}")
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
            print(f"Rate Limiter: Waiting {sleep_time:.2f} seconds for rate limit reset")
            time.sleep(sleep_time)
        else:
            print("Rate Limiter: No need to wait, rate limit has already reset")

    def __extract_rate_limit_info(self, response: requests.Response) -> tuple[Optional[int], Optional[float]]:
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
        if 'x-ratelimit-remaining' in response.headers:
            remaining_requests = int(response.headers['x-ratelimit-remaining'])
            print(f"Rate Limiter: Remaining requests: {remaining_requests}")
        
        if 'x-ratelimit-reset' in response.headers:
            reset_time = float(response.headers['x-ratelimit-reset'])
            print(f"Rate Limiter: Reset time: {reset_time}")
        elif 'x-ratelimit-timeremaining' in response.headers:
            # GitHub Models API uses time remaining in seconds
            time_remaining = int(response.headers['x-ratelimit-timeremaining'])
            reset_time = time.time() + time_remaining
            print(f"Rate Limiter: Time remaining: {time_remaining}s, Reset time: {reset_time}")
        
        #print(f"Rate Limiter: Available headers: {list(response.headers.keys())}")
        return remaining_requests, reset_time

    def make_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], 
                    timeout: int = 30, max_retries: int = 3) -> requests.Response:
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
        base_delay = 1.0
        reset_time: Optional[float] = None
        
        while retry_count <= max_retries:
            # Wait if we know we're rate limited
            if self.__should_wait_for_reset(reset_time):
                self.__wait_for_reset(reset_time)  # type: ignore
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                # Extract rate limit info from response
                _, new_reset_time = self.__extract_rate_limit_info(response)
                if new_reset_time:
                    reset_time = new_reset_time
                
                # Handle rate limit responses
                if response.status_code in (403, 429):
                    retry_after = response.headers.get('retry-after')
                    if retry_after:
                        sleep_time = int(retry_after)
                        print(f"Rate Limiter: Server requested retry after {sleep_time} seconds")
                    else:
                        # Exponential backoff
                        sleep_time = base_delay * (2 ** retry_count)
                        print(f"Rate Limiter: Using exponential backoff: {sleep_time} seconds")

                    retry_count += 1
                    if retry_count > max_retries:
                        print("Rate Limiter: Max retries exceeded, raising error")
                        response.raise_for_status()  # Raise the final error

                    print(f"Rate Limiter: Sleeping for {sleep_time} seconds before retry")
                    time.sleep(sleep_time)
                    continue
                  # Raise for other HTTP errors
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as exc:
                print(f"Rate Limiter: Request exception occurred: {exc}")
                if retry_count >= max_retries:
                    print("Rate Limiter: Max retries exceeded for exception, raising")
                    raise exc
                
                # Exponential backoff for other errors
                sleep_time = base_delay * (2 ** retry_count)
                print(f"Rate Limiter: Exception backoff: {sleep_time} seconds")
                retry_count += 1
                time.sleep(sleep_time)
        
        # This should never be reached, but included for safety
        raise requests.exceptions.RequestException("Max retries exceeded")
