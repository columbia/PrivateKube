from time import sleep

from stoppable_thread import StoppableThread, get_stop_flag
from privacy_resource_client import PrivacyResourceClient


class Snapshot:
    def __init__(self, client: PrivacyResourceClient):
        self.client = client
        pass

    def get_all_data_blocks(self, namespace):
        return self.client.list_data_blocks(namespace)

    def get_all_privacy_budget_claims(self, namespace):
        return self.client.list_privacy_budget_claims(namespace)

    def _get_all_data_blocks_periodically(
        self, namespace, period, data_block_handler=None
    ):
        if period < 1:
            period = 1

        stop_flag = get_stop_flag()

        if data_block_handler is None:
            data_block_handler = print

        while not stop_flag():
            data_blocks = self.get_all_data_blocks(namespace)
            data_block_handler(data_blocks)
            sleep(period)

    def get_all_data_blocks_periodically(self, *args, **kwargs):
        thread = StoppableThread(
            target=self._get_all_data_blocks_periodically, args=args, kwargs=kwargs
        )
        thread.start()
        return thread


if __name__ == "__main__":
    snapshot = Snapshot(PrivacyResourceClient())
    print(snapshot.get_all_data_blocks("privacy-example"))
    print(snapshot.get_all_privacy_budget_claims("privacy-example"))
    block_snapshot_thread = snapshot.get_all_data_blocks_periodically(
        "privacy-example", 3
    )
    sleep(15)
    block_snapshot_thread.stop()
