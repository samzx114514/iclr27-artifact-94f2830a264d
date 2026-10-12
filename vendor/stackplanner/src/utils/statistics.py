from typing import List


class GlobalStatistics:
    """
    A class to manage global statistics for the application.
    This class is designed
    """

    def __init__(self):
        self.agent_times: List[dict] = []

    def add_time_entry(self, data: dict):
        """
        Update the time taken by an agent for a specific task.
        """
        self.agent_times.append(data)

    def get_agent_times(self):
        """
        Get the list of agent times.
        """
        return self.agent_times

    def get_statistics(self):
        """
        Get a summary of the global statistics.
        """
        return {"agent_times": self.get_agent_times()}


global_statistics = GlobalStatistics()

import functools
from datetime import datetime
import asyncio


def timed_step(step_name):
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                time_entry = {
                    "step_name": step_name + start_time.isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
                global_statistics.add_time_entry(time_entry)

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                time_entry = {
                    "step_name": step_name + start_time.isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
                global_statistics.add_time_entry(time_entry)

        # 判断是否为异步函数
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
