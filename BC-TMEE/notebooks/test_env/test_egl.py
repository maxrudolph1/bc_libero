import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['DISPLAY'] = ''
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    import OpenGL.EGL as egl
    print("EGL library successfully imported")

    # Initialize EGL
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    if display == egl.EGL_NO_DISPLAY:
        print("Failed to get EGL display")
        exit(1)

    # Initialize EGL
    major, minor = egl.EGLint(), egl.EGLint()
    if not egl.eglInitialize(display, major, minor):
        print("EGL initialization failed")
        exit(1)

    print(f"EGL version: {major.value}.{minor.value}")

    # Get supported client APIs
    apis = egl.eglQueryString(display, egl.EGL_CLIENT_APIS)
    print("Supported client APIs:", apis)

    # Terminate EGL
    egl.eglTerminate(display)

except ImportError:
    print("Failed to import EGL library")
except Exception as e:
    print(f"Error occurred: {e}")
