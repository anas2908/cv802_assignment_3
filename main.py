import os
import sys

# On Apple silicon, PyTorch runs the novel-view renderers on the GPU (MPS); the few operations MPS lacks fall back to
# the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import open3d.visualization.gui as gui  # noqa: E402

from modules.gui.gui import AppWindow  # noqa: E402


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1280, 860)

    # python main.py results          -> novel-view results folder (has scene.json)
    # python main.py <mesh.obj|.ply>  -> a mesh
    # python main.py <colmap folder>  -> a precomputed COLMAP result (Sub-project 1)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(os.path.join(path, "scene.json")):
            w.load_nvs_results(path)
        elif os.path.isfile(path) and path.lower().endswith((".obj", ".ply", ".glb", ".gltf")):
            w.load_mesh(path)
        elif os.path.exists(path):
            w.load_existing_result(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
