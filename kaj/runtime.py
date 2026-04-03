from multiprocessing import get_context
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

import config
from verifier import Verifier
from controller import Controller
# from controller_rgi import Controller
from animation import Animation


class Runtime:
    def __init__(self):
        self.ctx = get_context("spawn")
        self.manager: SyncManager = self.ctx.Manager()

        self.registry = self.manager.dict()
        self.registry["_active"] = {}

        self.state_queue = self.ctx.Queue()
        self.stop_event = self.ctx.Event()

        self.p_verifier = None
        self.p_controller = None
        self.p_animation = None

    # ------------------------------------------------------------------
    # Process entrypoints
    # ------------------------------------------------------------------

    @staticmethod
    def _verifier_main(registry, state_queue, stop_event, run_args):
        verifier = Verifier(registry, state_queue, stop_event)
        try:
            verifier.run(*run_args)
        finally:
            verifier.close()

    @staticmethod
    def _controller_main(registry, state_queue, stop_event, run_args):
        controller = Controller(registry, state_queue, stop_event)
        try:
            controller.run(*run_args)
        finally:
            controller.close()

    @staticmethod
    def _animation_main(registry, state_queue, stop_event, run_args):
        animation = Animation(registry, state_queue, stop_event)
        try:
            animation.run(*run_args)
        finally:
            animation.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, verifier_args=(), controller_args=(), animation_args=()):
        self.p_verifier = self.ctx.Process(
            target=self._verifier_main,
            args=(self.registry, self.state_queue, self.stop_event, verifier_args),
            name="VerifierProcess",
        )
        self.p_controller = self.ctx.Process(
            target=self._controller_main,
            args=(self.registry, self.state_queue, self.stop_event, controller_args),
            name="ControllerProcess",
        )
        self.p_animation = self.ctx.Process(
            target=self._animation_main,
            args=(self.registry, self.state_queue, self.stop_event, animation_args),
            name="AnimationProcess",
        )

        self.p_verifier.start()
        self.p_controller.start()
        self.p_animation.start()

    def wait(self):
        if self.p_controller is not None:
            self.p_controller.join()
        if self.p_animation is not None:   
            self.p_animation.join()

    def stop(self):
        self.stop_event.set()
        self.state_queue.put(None)   # sentinel for verifier if blocked on queue

        if self.p_verifier is not None:
            self.p_verifier.join(timeout=5.0)
            if self.p_verifier.is_alive():
                self.p_verifier.terminate()
                self.p_verifier.join()

        if self.p_controller is not None and self.p_controller.is_alive():
            self.p_controller.terminate()
            self.p_controller.join()

        if self.p_animation is not None and self.p_animation.is_alive():
            self.p_animation.terminate()
            self.p_animation.join()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_registry(self):
        active = dict(self.registry.get("_active", {}))
        for _, meta in active.items():
            try:
                shm = SharedMemory(name=meta["shm_name"])
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        self.registry["_active"] = {}

    def close(self):
        self.cleanup_registry()
        self.manager.shutdown()

    # ------------------------------------------------------------------
    # Full run helper
    # ------------------------------------------------------------------

    def run(self):

        from math import pi

        # Write shared configuration into registry before any subprocess starts
        config.run(self.registry)

        t0 = 0.0
        x0 = [3.0, 0.5, pi/2, 0.0]

        try:
            self.start(
                verifier_args=(),
                controller_args=(t0, x0),
                animation_args=(),
            )
            self.wait()
        finally:
            self.stop()
            self.close()


if __name__ == "__main__":
    Runtime().run()