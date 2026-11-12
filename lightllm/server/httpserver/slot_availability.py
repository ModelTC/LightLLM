import asyncio


class RequestSlotAvailability:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.generation = 0
        self.event = asyncio.Event() if enabled else None

    @classmethod
    def from_args(cls, args):
        return cls(
            enabled=(
                args.run_mode == "normal"
                and args.dp == 1
                and args.nnodes == 1
                and not args.enable_rl
                and args.httpserver_workers == 1
            )
        )

    def notify_release(self):
        if self.enabled:
            self.generation += 1
            self.event.set()

    async def wait(self, observed_generation, timeout):
        if not self.enabled:
            await asyncio.sleep(timeout)
            return False
        if observed_generation != self.generation:
            return True
        self.event.clear()
        try:
            await asyncio.wait_for(self.event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
