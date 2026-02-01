## Error Catching

- RuntimeError: Fail to initialize OpenGL: enter the following command in the terminal
```bash
unset LD_PRELOAD
```

- If libgpu_partition.so confilts with gym and robosuite libraries
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0
```

