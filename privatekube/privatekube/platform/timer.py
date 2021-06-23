import time


class Timer:
    def __init__(self, speed=1):
        self.offset = time.time()
        self.speed = speed

    def current(self):
        """
        Return the the relative timestamp in ms.
        :return: timestamp in ms.
        """
        cur = time.time()
        return (cur - self.offset) * self.speed

    def wait_until(self, until_time):
        interval = until_time - self.current()
        if interval <= 0:
            return
        time.sleep(interval / self.speed)

    def sleep(self, sleep_time):
        time.sleep(sleep_time / self.speed)

    def duration_in_k8s(self, duration):
        # 1000 is to convert second to millisecond
        return duration / self.speed * 1000
