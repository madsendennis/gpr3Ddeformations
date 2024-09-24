import numpy as np
import pyrender
import ipywidgets as widgets
from IPython.display import display
import queue
import threading
import time
from pyrender.constants import DEFAULT_Z_FAR, DEFAULT_Z_NEAR
from gpmm import GPMM
from trimesh import Trimesh
from pyrender.material import Material


class PyGpmmUI:
    def __init__(self):
        self.scene = pyrender.Scene()
        self.viewer = pyrender.Viewer(
            scene=self.scene,
            use_raymond_lighting=True,
            run_in_thread=True,
            lighting_intensity=5.0,
        )
        self.update_queue = queue.Queue()
        self.gpmm_controls = {}  # Dictionary to store controls for each GPMM
        self.mesh_nodes = {}  # Dictionary to store mesh nodes for each GPMM
        # Start the update worker thread
        threading.Thread(target=self._update_mesh_worker, daemon=True).start()
        self.add_reset_camera_button()

    def add_gpmm(self, gpmm: GPMM, name: str, num_components: int = 3):
        rank = gpmm.rank
        num_components = min(num_components, rank)

        # Create UI elements
        sliders = [
            widgets.FloatSlider(min=-3, max=3, step=0.1, description=f"PC {i+1}")
            for i in range(num_components)
        ]
        random_button = widgets.Button(description="Random")
        zero_button = widgets.Button(description="Zero")

        # Create initial mesh and add to scene
        initial_z = np.zeros(rank)
        mesh_tri = self._create_mesh(gpmm, initial_z)
        mesh_node, material = self._add_mesh_to_scene(mesh_tri)

        self.mesh_nodes[name] = mesh_node

        def queue_update(_):
            z = np.zeros(rank)
            for i, slider in enumerate(sliders):
                z[i] = slider.value
            self._empty_and_add_to_queue((name, z))

        def set_random_values(_):
            z = np.clip(np.random.normal(0, 1, rank), -3, 3)
            self._empty_and_add_to_queue((name, z))

        def set_zero_values(_):
            self._empty_and_add_to_queue((name, np.zeros(rank)))

        # Connect UI elements to functions
        for slider in sliders:
            slider.observe(queue_update, names="value")
        random_button.on_click(set_random_values)
        zero_button.on_click(set_zero_values)

        # Store controls
        self.gpmm_controls[name] = {
            "gpmm": gpmm,
            "sliders": sliders,
            "random_button": random_button,
            "zero_button": zero_button,
            "material": material,
        }

        # Create and display control panel
        control_panel = widgets.VBox(
            [widgets.HTML(f"<b>{name}</b>"), random_button, zero_button] + sliders
        )
        display(control_panel)
        self.reset_camera()

    def _create_mesh(self, gpmm: GPMM, z: np.array):
        return gpmm.instance(z)

    def _add_mesh_to_scene(self, mesh_tri: Trimesh):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            roughnessFactor=0.7,
            baseColorFactor=(0.8, 0.8, 0.8, 1.0),
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_tri, material=material)
        mesh_node = self.scene.add(pyrender_mesh)
        return mesh_node, material

    def _update_mesh_worker(self):
        while True:
            try:
                name, z = self.update_queue.get(timeout=1.0)
                if name in self.gpmm_controls:
                    controls = self.gpmm_controls[name]
                    self._update_sliders(controls["sliders"], z)
                    self._update_scene(name, controls["gpmm"], z, controls["material"])
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in update_mesh_worker: {e}")

    def _update_sliders(self, sliders, z):
        for i, slider in enumerate(sliders):
            if i < len(z):
                slider.value = z[i]

    def _update_scene(self, name: str, gpmm: GPMM, z: np.ndarray, material: Material):
        mesh_tri = self._create_mesh(gpmm, z)
        new_pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_tri, material=material)

        if self.viewer.is_active:
            with self.viewer.render_lock:
                old_mesh_node = self.mesh_nodes[name]
                if old_mesh_node in self.scene.nodes:
                    self.scene.remove_node(old_mesh_node)
                new_mesh_node = self.scene.add(new_pyrender_mesh)
                self.mesh_nodes[name] = new_mesh_node

    def _empty_and_add_to_queue(self, item: np.ndarray):
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()
            except queue.Empty:
                break
        self.update_queue.put(item)

    def reset_camera(self, _=None):
        if self.viewer.is_active:
            # Calculation of zfar and znear from pyrender.viewer __init__
            zfar = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
            if self.scene.scale == 0:
                znear = DEFAULT_Z_NEAR
            else:
                znear = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)

            with self.viewer.render_lock:
                self.scene.main_camera_node.camera.zfar = zfar
                self.scene.main_camera_node.camera.znear = znear

                self.viewer._default_camera_pose = (
                    self.viewer._compute_initial_camera_pose()
                )
                self.viewer._reset_view()

    def add_reset_camera_button(self):
        reset_button = widgets.Button(description="Reset Camera")
        reset_button.on_click(self.reset_camera)
        display(reset_button)
