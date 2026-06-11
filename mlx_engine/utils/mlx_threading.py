import mlx.core as mx


def install_mlx_compile_cache_cleanup_for_thread() -> None:
    """Install MLX's compile-cache cleanup guard on the current thread."""
    # MLX 0.31.2 has a bug where a thread can call an mx.compile-created
    # function without installing the thread-local compiler-cache cleanup guard.
    # If that cache later releases Python output metadata during thread teardown,
    # it can do so without the GIL and crash the process. MLX main has fixed this
    # for 0.31.3 or greater by installing cleanup when compiled functions run.
    mx.compile(lambda x: x)
