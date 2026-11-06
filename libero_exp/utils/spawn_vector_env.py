"""LIBERO subprocess workers with independent EGL/CUDA process state."""
import multiprocessing as mp

from libero.libero.envs.venv import (
    BaseVectorEnv, CloudpickleWrapper, EnvWorker, SubprocEnvWorker,
    SubprocVectorEnv, _worker,
)


class SpawnEnvWorker(SubprocEnvWorker):
    def __init__(self, env_fn):
        # LIBERO's worker fixes multiprocessing.Process to the default context.
        # Retain its RPC protocol, but create pipes/processes in spawn context
        # so initialized EGL and CUDA state cannot be inherited through fork.
        context = mp.get_context("spawn")
        self.parent_remote, self.child_remote = context.Pipe()
        self.share_memory = False
        self.buffer = None
        self.process = context.Process(
            target=_worker,
            args=(self.parent_remote, self.child_remote, CloudpickleWrapper(env_fn), None),
            daemon=True,
        )
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)


class SpawnSubprocVectorEnv(SubprocVectorEnv):
    def __init__(self, env_fns, **kwargs):
        BaseVectorEnv.__init__(self, env_fns, SpawnEnvWorker, **kwargs)
