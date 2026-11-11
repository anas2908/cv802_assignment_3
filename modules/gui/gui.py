# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import glob
import time
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import os.path as osp
import platform
import sys
import threading

from modules.colmap.api import ColmapAPI
from modules.gui.settings import Settings
from utils.thread_utils import run_on_thread

isMacOS = (platform.system() == "Darwin")


class AppWindow:
    MENU_OPEN_EXISTING = 11
    MENU_OPEN_IMAGE_FOLDER = 12
    MENU_OPEN_VIDEO = 13
    MENU_EXPORT = 14
    MENU_QUIT = 15
    MENU_OPEN_NVS = 16
    MENU_OPEN_MESH = 17
    MENU_EXPORT_VIDEO = 18
    MENU_SHOW_SETTINGS = 21
    MENU_ABOUT = 31

    DEFAULT_IBL = "default"

    def __init__(self, width, height):
        self.settings = Settings()

        # COLMAP API
        self.colmap_api = ColmapAPI(
            gpu_index=self.settings.DEFAULT_GPU_INDEX,
            camera_model=self.settings.DEFAULT_CAMERA_MODEL,
            matcher=self.settings.DEFAULT_COLMAP_MATCHER,
        )

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # Sizing
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Setting panel
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # GUI Control
        gui_ctrls = gui.CollapsableVert(
            "GUI settings", 0.25 * em,
            gui.Margins(em, 0, 0, 0)
        )
        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        gui_ctrls.add_child(grid)

        self._camera_color = gui.ColorEdit()
        self._camera_color.set_on_value_changed(self._on_camera_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Cam Color"))
        grid.add_child(self._camera_color)
        gui_ctrls.add_child(grid)

        self._camera_size = gui.Slider(gui.Slider.DOUBLE)
        self._camera_size.set_limits(0.1, 1)
        self._camera_size.set_on_value_changed(self._on_camera_size)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Cam Size"))
        grid.add_child(self._camera_size)
        grid.add_child(gui.Label("Point Size"))
        grid.add_child(self._point_size)
        gui_ctrls.add_child(grid)

        self._settings_panel.add_child(gui_ctrls)

        # COLMAP Control
        colmap_ctrls = gui.CollapsableVert("COLMAP settings", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._camera_models = gui.Combobox()
        for name in Settings.CAMERA_MODELS:
            self._camera_models.add_item(name)
        self._camera_models.set_on_selection_changed(self._on_colmap_camera_model_change)
        
        self._colmap_matchers = gui.Combobox()
        for name in Settings.COLMAP_MATCHERS:
            self._colmap_matchers.add_item(name)
        self._colmap_matchers.set_on_selection_changed(self._on_colmap_matcher_change)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Camera"))
        grid.add_child(self._camera_models)
        grid.add_child(gui.Label("Matchers"))
        grid.add_child(self._colmap_matchers)
        colmap_ctrls.add_child(grid)

        h = gui.Horiz(0.25 * em)  # row 2
        self._fit_colmap_button = gui.Button("Fit Colmap")
        self._fit_colmap_button.horizontal_padding_em = 0.2
        self._fit_colmap_button.enabled = False
        self._fit_colmap_button.set_on_clicked(self._on_fit_colmap_button)
        h.add_stretch()
        h.add_child(self._fit_colmap_button)
        h.add_stretch()
        colmap_ctrls.add_child(h)

        self.colmap_ctrls = colmap_ctrls
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.colmap_ctrls)

        # Novel-view synthesis and meshes (Sub-projects 2 and 3)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self._build_nvs_panel(em))
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self._build_mesh_panel(em))
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)

            file_menu = gui.Menu()
            file_menu.add_item("Open existing result...", AppWindow.MENU_OPEN_EXISTING)
            file_menu.add_item("Open image folder...", AppWindow.MENU_OPEN_IMAGE_FOLDER)
            file_menu.add_item("Open novel-view results...", AppWindow.MENU_OPEN_NVS)
            file_menu.add_item("Open mesh...", AppWindow.MENU_OPEN_MESH)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            file_menu.add_item("Export novel-view video...", AppWindow.MENU_EXPORT_VIDEO)

            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)

            settings_menu = gui.Menu()
            settings_menu.add_item("3D Reconstruction",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN_EXISTING, self._on_menu_open_existing)
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN_IMAGE_FOLDER, self._on_menu_open_image_folder)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN_NVS, self._on_menu_open_nvs)
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN_MESH, self._on_menu_open_mesh)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT_VIDEO, self._on_nvs_export)
        # ----

        # Live novel views: on every tick, re-render when the camera has moved (see _on_tick)
        w.set_on_tick_event(self._on_tick)

        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_camera_size(self, size):
        self.settings.camera_size = size
        self.settings.apply_camera = True
        self._apply_settings()

    def _on_fit_colmap_button(self):
        # Fitting should honor the currently selected camera model and
        # matcher. "Open existing result" remains the cache-only path.
        self.colmap_api.estimate_cameras(recompute=True)

        em = self.window.theme.font_size
        dlg = gui.Dialog("COLMAP")

        # Add the text
        dlg_layout = gui.Vert(em / 2, gui.Margins(em, em, em, em))
        self._colmap_running_label = gui.Label("Running COLMAP. Please wait ...")
        dlg_layout.add_child(self._colmap_running_label)

        self._fit_colmap_ok_button = gui.Button("Close")
        self._fit_colmap_ok_button.set_on_clicked(self._on_info_ok)
        self._fit_colmap_ok_button.enabled = False

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(self._fit_colmap_ok_button)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

        self._enable_colmap_ok_button_when_done()

    def _enable_colmap_ok_button_when_done(self):
        w = self.window
        if self.colmap_api.estimate_done():
            if self.colmap_api.estimate_error is None:
                self._add_geometries_from_colmap()
                self._colmap_running_label.text = 'Done!'
            else:
                self._colmap_running_label.text = (
                    'COLMAP failed: ' + str(self.colmap_api.estimate_error)
                )
            self._fit_colmap_ok_button.enabled = True
            w.post_redraw()
        else:
            gui.Application.instance.post_to_main_thread(w, self._enable_colmap_ok_button_when_done)

    def _on_colmap_matcher_change(self, name, index):
        self.colmap_api.matcher = name

    def _on_colmap_camera_model_change(self, name, index):
        self.colmap_api.camera_model = name

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)

        self._bg_color.color_value = self.settings.bg_color
        self._camera_color.color_value = self.settings.camera_color

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        if self.settings.apply_camera:
            self._visualize_cameras()
            self.settings.apply_camera = False

        self._point_size.double_value = self.settings.material.point_size
        self._camera_size.double_value = self.settings.camera_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_camera_color(self, new_color):
        self.settings.camera_color = new_color
        self.settings.apply_camera = True
        self._apply_settings()

    def _on_menu_open_existing(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR,
            "Choose result folder to load",
            self.window.theme
        )

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_existing_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_open_image_folder(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN_DIR,
            "Choose image folder to load",
            self.window.theme
        )

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_image_folder_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_existing_dialog_done(self, filename):
        self.window.close_dialog()
        self.load_existing_result(filename)

    def _on_load_image_folder_dialog_done(self, data_path):
        self.window.close_dialog()

        self.colmap_api.data_path = data_path

        # Verify if there is any image in this folder
        if len(self.colmap_api._list_images_in_folder(self.colmap_api.image_dir)) == 0:
            self.colmap_api.data_path = None
            em = self.window.theme.font_size
            dlg = gui.Dialog("Error")

            # Add the text
            dlg_layout = gui.Vert(em / 2, gui.Margins(em, em, em, em))
            dlg_layout.add_child(gui.Label("There is no image in this folder! Choose another one."))

            ok = gui.Button("OK")
            ok.set_on_clicked(self._on_error_ok)

            h = gui.Horiz()
            h.add_stretch()
            h.add_child(ok)
            h.add_stretch()
            dlg_layout.add_child(h)

            dlg.add_child(dlg_layout)
            self.window.show_dialog(dlg)

        self._fit_colmap_button.enabled = True

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em / 2, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Code framework for the lecture: CV802: Advanced 3D Computer Vision"))
        dlg_layout.add_child(gui.Label("Instructor: Hao Li"))
        dlg_layout.add_child(gui.Label("Co-Instructure: Paul"))
        dlg_layout.add_child(gui.Label("TAs: Phong Tran, Nhat Ho"))
        dlg_layout.add_child(gui.Label("Copyright (C) 2024 by Metaverse Lab, MBZUAI"))
        dlg_layout.add_child(gui.Label("License:"))
        dlg_layout.add_child(gui.Label(
            (
                "This program is free software; you can redistribute it and/or modify "
                "it under the terms of the GNU General Public Licenseas published by the Free Software Foundation; either version 2 "
                "of the License, or (at your option) any later version."
            )
        ))
        dlg_layout.add_child(gui.Label(
            (
                "This program is distributed in the hope that it will be useful, "
                "but WITHOUT ANY WARRANTY; without even the implied warranty of "
                "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "
                "GNU General Public License for more details."
            )
        ))
        dlg_layout.add_child(gui.Label(
            (
                "You should have received a copy of the GNU General Public License "
                "along with this program; if not, write to the Free Software "
                "Foundation, Inc., 51 Franklin Street, Fifth Floor, "
                "Boston, MA  02110-1301, USA."
            )
        ))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _on_error_ok(self):
        self.window.close_dialog()

    def _on_info_ok(self):
        self.window.close_dialog()

    def _on_camera_list_change(self, name, index):
        self.colmap_api.activate_camera_name = name
        self._update_camera()

    # ------------------------------------------------------------------ Sub-project 3: novel-view synthesis
    # A trained model (3DGS, Mip-Splatting or TensoRF) is drawn from the 3D view's own camera: drag, pan or zoom
    # as usual and the novel view follows. While the camera moves a small preview is rendered; once it stops,
    # the view is rendered again at the final size. Renders run on a worker thread (CUDA, Apple MPS or CPU) and
    # are shown as the scene background, so the usual mouse controls keep working.
    def _build_nvs_panel(self, em):
        self._nvs = None
        self._nvs_mid = None
        self._nvs_key = None
        self._nvs_level = 0          # 0 = needs a render, 1 = preview shown, 2 = final shown
        self._nvs_moved = 0.0
        self._nvs_busy = False
        self._nvs_keys = []
        self._nvs_geoms = []

        ctrls = gui.CollapsableVert("Novel-view synthesis", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._nvs_method = gui.Combobox()
        self._nvs_method.set_on_selection_changed(self._on_nvs_method)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Method"))
        grid.add_child(self._nvs_method)
        ctrls.add_child(grid)

        self._nvs_live = gui.Checkbox("Live novel view (drag to move)")
        self._nvs_live.set_on_checked(self._on_nvs_live)
        ctrls.add_child(self._nvs_live)

        self._nvs_preview = gui.Slider(gui.Slider.DOUBLE)
        self._nvs_preview.set_limits(0.1, 1.0)
        self._nvs_preview.double_value = 0.3
        self._nvs_final = gui.Slider(gui.Slider.DOUBLE)
        self._nvs_final.set_limits(0.25, 1.0)
        self._nvs_final.double_value = 0.8
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Moving"))
        grid.add_child(self._nvs_preview)
        grid.add_child(gui.Label("Still"))
        grid.add_child(self._nvs_final)
        ctrls.add_child(grid)

        self._nvs_photos = gui.Combobox()
        self._nvs_photos.set_on_selection_changed(self._on_nvs_photo_selected)
        self._nvs_show_photo = gui.Checkbox("Show the real photo")
        self._nvs_show_photo.set_on_checked(self._on_nvs_show_photo)
        ctrls.add_child(gui.Label("Held-out photos (not used in training)"))
        ctrls.add_child(self._nvs_photos)
        ctrls.add_child(self._nvs_show_photo)

        h = gui.Horiz(0.25 * em)
        add_key = gui.Button("Add keyframe")
        add_key.horizontal_padding_em = 0.2
        add_key.set_on_clicked(self._on_nvs_add_key)
        clear_key = gui.Button("Clear")
        clear_key.horizontal_padding_em = 0.2
        clear_key.set_on_clicked(self._on_nvs_clear_keys)
        h.add_child(add_key)
        h.add_child(clear_key)
        ctrls.add_child(h)
        self._nvs_path = gui.Combobox()
        self._nvs_path.add_item("Preset orbit (8 s)")
        self._nvs_path.add_item("My keyframes")
        self._nvs_seconds = gui.Slider(gui.Slider.INT)
        self._nvs_seconds.set_limits(2, 20)
        self._nvs_seconds.int_value = 8
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Path"))
        grid.add_child(self._nvs_path)
        grid.add_child(gui.Label("Seconds"))
        grid.add_child(self._nvs_seconds)
        ctrls.add_child(grid)
        export = gui.Button("Export MP4...")
        export.horizontal_padding_em = 0.2
        export.set_on_clicked(self._on_nvs_export)
        ctrls.add_child(export)

        self._nvs_status = gui.Label("File > Open novel-view results...")
        ctrls.add_child(self._nvs_status)
        return ctrls

    def _build_mesh_panel(self, em):
        ctrls = gui.CollapsableVert("Meshes (Sub-project 2)", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._mesh_list = gui.Combobox()
        self._mesh_list.set_on_selection_changed(self._on_mesh_selected)
        self._mesh_paths = []
        self._mesh_show = gui.Checkbox("Show mesh")
        self._mesh_show.set_on_checked(self._on_mesh_show)
        ctrls.add_child(self._mesh_list)
        ctrls.add_child(self._mesh_show)
        ctrls.add_child(gui.Label("Textured triangles, rasterised in this view"))

        # Sub-project 2 in the app: our Screened Poisson on the loaded point cloud
        self._extract_res = gui.Combobox()
        for r in ("128^3 grid (fast)", "256^3 grid (detailed)"):
            self._extract_res.add_item(r)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Poisson"))
        grid.add_child(self._extract_res)
        ctrls.add_child(grid)
        h = gui.Horiz(0.25 * em)
        extract = gui.Button("Extract mesh")
        extract.horizontal_padding_em = 0.2
        extract.set_on_clicked(self._on_extract_mesh)
        save = gui.Button("Save mesh...")
        save.horizontal_padding_em = 0.2
        save.set_on_clicked(self._on_save_mesh)
        h.add_child(extract)
        h.add_child(save)
        ctrls.add_child(h)
        self._extract_status = gui.Label("from the open point cloud (Sub-project 1)")
        ctrls.add_child(self._extract_status)
        self._extracted = None
        return ctrls

    def _points_for_mesh(self):
        """The open point cloud: a COLMAP result of Sub-project 1 if one is loaded, else the novel-view results' cloud."""
        try:
            pcd = self.colmap_api.pcd if self.colmap_api.estimate_done() else None
            if pcd is not None and len(pcd.points) > 0:
                return np.asarray(pcd.points), np.asarray(pcd.colors), "COLMAP result"
        except Exception:
            pass
        if self._nvs is not None:
            pcd = o3d.io.read_point_cloud(str(self._nvs.root / self._nvs.meta["points"]))
            return np.asarray(pcd.points), np.asarray(pcd.colors), "results point cloud"
        return None

    def _on_extract_mesh(self):
        pts = self._points_for_mesh()
        if pts is None:
            self._message("Extract mesh", "Open a COLMAP result or the novel-view results first (File menu).")
            return
        xyz, rgb, source = pts
        res = 128 if self._extract_res.selected_index == 0 else 256
        self._extract_status.text = "Poisson on %d points (%s), %d^3 ..." % (len(xyz), source, res)

        def job():
            try:
                from modules.surface.extract import extract_mesh
                V, F, C, info = extract_mesh(xyz, rgb, res=res, log=lambda *a: None)
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
                mesh.compute_vertex_normals()
                err = None
            except Exception as error:
                mesh, info, err = None, None, error

            def done():
                if err is not None:
                    self._extract_status.text = "Poisson failed: %s" % err
                    return
                self._extracted = mesh
                label = "Poisson %d^3 (extracted here)" % res
                self._mesh_list.add_item(label)
                self._mesh_paths.append(mesh)
                self._mesh_list.selected_index = len(self._mesh_paths) - 1
                self.load_mesh(mesh)
                self._extract_status.text = "%d triangles in %.0f s" % (info["triangles"], info["seconds"])
            gui.Application.instance.post_to_main_thread(self.window, done)
        threading.Thread(target=job, daemon=True).start()

    def _on_save_mesh(self):
        if self._extracted is None:
            self._message("Save mesh", "Extract a mesh first.")
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Save the mesh as", self.window.theme)
        dlg.add_filter(".ply", "PLY mesh (.ply)")
        dlg.add_filter(".obj", "OBJ mesh (.obj)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_save_mesh_done)
        self.window.show_dialog(dlg)

    def _on_save_mesh_done(self, path):
        self.window.close_dialog()
        if not path.lower().endswith((".ply", ".obj")):
            path += ".ply"
        ok = o3d.io.write_triangle_mesh(path, self._extracted)
        self._extract_status.text = ("saved " if ok else "could not save ") + osp.basename(path)

    def _message(self, title, text):
        self.window.show_message_box(title, text)

    def _on_menu_open_nvs(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose a novel-view results folder", self.window.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_nvs_dialog_done)
        self.window.show_dialog(dlg)

    def _on_nvs_dialog_done(self, folder):
        self.window.close_dialog()
        self.load_nvs_results(folder)

    def load_nvs_results(self, folder):
        try:
            from modules.nvs.session import NVSSession
            self._nvs = NVSSession(folder)
        except Exception as error:
            self._nvs = None
            self._message("Error", "Not a novel-view results folder: %s" % error)
            return
        s = self._nvs
        self._nvs_method.clear_items()
        self._nvs_ids = []
        for mid, label in s.labels():
            self._nvs_method.add_item(label)
            self._nvs_ids.append(mid)
        first = self._nvs_ids.index("3dgs") if "3dgs" in self._nvs_ids else 0     # start on a real-time method
        self._nvs_method.selected_index = first
        self._nvs_mid = self._nvs_ids[first]

        # the person's points and the photo cameras, for orientation (hidden while the live view is on)
        for name in self._nvs_geoms:
            self._scene.scene.remove_geometry(name)
        self._nvs_geoms = []
        pts = s.root / s.meta["points"]
        if pts.exists():
            self._scene.scene.add_geometry("__nvs_points__", o3d.io.read_point_cloud(str(pts)), self.settings.material)
            self._nvs_geoms.append("__nvs_points__")
        test = set(s.meta["split"]["test"])
        for i, c in enumerate(s.cameras):
            lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=int(c.W), view_height_px=int(c.H),
                intrinsic=np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]]),
                extrinsic=self._extrinsic(c), scale=0.15)
            colour = [0.95, 0.45, 0.1] if c.name in test else [0.2, 0.45, 0.85]
            lines.colors = o3d.utility.Vector3dVector([colour] * len(lines.lines))
            self._scene.scene.add_geometry("__nvs_cam%d__" % i, lines, self.settings.material)
            self._nvs_geoms.append("__nvs_cam%d__" % i)

        self._nvs_photos.clear_items()
        self._nvs_test = [c for c in s.cameras if c.name in test]
        for c in self._nvs_test:
            self._nvs_photos.add_item(c.name.split("__")[-1])

        self._mesh_list.clear_items()
        self._mesh_paths = []
        for m in s.meta.get("meshes", []):
            self._mesh_list.add_item(m["label"])
            self._mesh_paths.append(str(s.root / m["file"]))

        self._snap_to(self._nvs_test[0] if self._nvs_test else s.cameras[0])
        self._nvs_live.checked = True
        self._on_nvs_live(True)
        self._nvs_status.text = "%d models · device %s" % (len(self._nvs_ids), s.device.type)
        self.window.set_needs_layout()

    @staticmethod
    def _extrinsic(c):
        e = np.eye(4)
        e[:3, :3], e[:3, 3] = c.R, c.t
        return e

    def _snap_to(self, c):
        intr = o3d.camera.PinholeCameraIntrinsic(int(c.W), int(c.H), c.fx, c.fy, c.cx, c.cy)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(intr, self._extrinsic(c), bounds)
        self._nvs_key = None

    def _view_camera(self, scale=1.0, max_side=None):
        """The 3D view's current camera as a pinhole camera in OpenCV axes (Open3D looks down -z with y up)."""
        from modules.nvs.cameras import Camera
        r = self._scene.frame
        W, H = max(int(r.width), 8), max(int(r.height), 8)
        cam = self._scene.scene.camera
        V = np.asarray(cam.get_view_matrix(), np.float64)
        flip = np.diag([1.0, -1.0, -1.0])
        fov = np.deg2rad(cam.get_field_of_view())
        if cam.get_field_of_view_type() == rendering.Camera.FovType.Horizontal:
            f = W / (2 * np.tan(fov / 2))
        else:
            f = H / (2 * np.tan(fov / 2))
        if max_side:
            scale = min(scale, max_side / max(W, H))
        return Camera(W, H, f, f, flip @ V[:3, :3], flip @ V[:3, 3]).scaled(scale)

    def _on_nvs_method(self, name, index):
        if self._nvs is None:
            return
        self._nvs_mid = self._nvs_ids[index]
        self._nvs_key = None
        self._nvs_status.text = "loading %s ..." % name

    def _on_nvs_live(self, on):
        for name in self._nvs_geoms + ["__model__", "__mesh__"]:
            if self._scene.scene.has_geometry(name):
                self._scene.scene.show_geometry(name, not on)
        if not on:
            self._apply_settings()
        self._nvs_key = None
        self.window.post_redraw()

    def _on_tick(self):
        if self._nvs is None or not self._nvs_live.checked or self._nvs_show_photo.checked:
            return False
        key = (tuple(np.round(np.asarray(self._scene.scene.camera.get_view_matrix()).ravel(), 5)),
               self._scene.frame.width, self._scene.frame.height, self._nvs_mid)
        now = time.time()
        if key != self._nvs_key:
            self._nvs_key, self._nvs_moved, self._nvs_level = key, now, 0
        if self._nvs_busy:
            return False
        if self._nvs_level == 0:
            self._nvs_render(self._nvs_preview.double_value, 1)
        elif self._nvs_level == 1 and now - self._nvs_moved > 0.35:
            self._nvs_render(self._nvs_final.double_value, 2)
        return False

    def _nvs_render(self, scale, level):
        cam, key, mid = self._view_camera(scale, max_side=1400), self._nvs_key, self._nvs_mid
        self._nvs_busy = True

        def job():
            try:
                img, dt = self._nvs.render(mid, cam)
                err = None
            except Exception as error:
                img, dt, err = None, 0.0, error

            def done():
                self._nvs_busy = False
                if err is not None:
                    self._nvs_level = 3
                    self._nvs_status.text = "render failed: %s" % err
                    return
                if key == self._nvs_key:
                    self._nvs_level = level
                if self._nvs_live.checked and not self._nvs_show_photo.checked:
                    self._scene.scene.set_background([1, 1, 1, 1], o3d.geometry.Image(img))
                self._nvs_status.text = "%s · %dx%d · %.2f s · %s" % (
                    self._nvs.models[mid]["label"], img.shape[1], img.shape[0], dt, self._nvs.device.type)
                self.window.post_redraw()
            gui.Application.instance.post_to_main_thread(self.window, done)
        threading.Thread(target=job, daemon=True).start()

    def _on_nvs_photo_selected(self, name, index):
        if self._nvs is not None and index < len(self._nvs_test):
            self._snap_to(self._nvs_test[index])
            if self._nvs_show_photo.checked:
                self._on_nvs_show_photo(True)

    def _on_nvs_show_photo(self, on):
        if self._nvs is None:
            return
        if on:
            c = self._nvs_test[max(0, self._nvs_photos.selected_index)]
            self._snap_to(c)
            p = self._nvs.photo(c.name)
            if p is not None:
                self._scene.scene.set_background([1, 1, 1, 1], o3d.io.read_image(str(p)))
                self._nvs_status.text = "real photo %s (held out)" % c.name.split("__")[-1]
        self._nvs_key = None
        self.window.post_redraw()

    def _on_nvs_add_key(self):
        cam = self._view_camera(1.0, max_side=960)
        if self._nvs_keys:            # every keyframe at the first one's size, same vertical field of view
            k0 = self._nvs_keys[0]
            f = k0.H / cam.H * cam.fy
            from modules.nvs.cameras import Camera
            cam = Camera(k0.W, k0.H, f, f, cam.R, cam.t)
        self._nvs_keys.append(cam)
        self._nvs_path.selected_index = 1
        self._nvs_status.text = "%d keyframes" % len(self._nvs_keys)

    def _on_nvs_clear_keys(self):
        self._nvs_keys = []
        self._nvs_status.text = "keyframes cleared"

    def _on_nvs_export(self):
        if self._nvs is None:
            self._message("Export", "Open novel-view results first (File menu).")
            return
        if self._nvs_path.selected_index == 1 and len(self._nvs_keys) < 2:
            self._message("Export", "Add at least two keyframes, or choose the preset orbit.")
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Save the video as", self.window.theme)
        dlg.add_filter(".mp4", "MP4 video (.mp4)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_nvs_export_done)
        self.window.show_dialog(dlg)

    def _on_nvs_export_done(self, path):
        self.window.close_dialog()
        if not path.endswith(".mp4"):
            path += ".mp4"
        n = 30 * int(self._nvs_seconds.int_value)
        if self._nvs_path.selected_index == 0:
            orbit = self._nvs.orbit()
            cams = [orbit[int(i * len(orbit) / n)] for i in range(n)]
        else:
            cams = self._nvs.path_from_keyframes(self._nvs_keys, n)
        mid = self._nvs_mid

        def progress(i, total):
            def show():
                self._nvs_status.text = "exporting %d / %d frames" % (i, total)
            gui.Application.instance.post_to_main_thread(self.window, show)

        def job():
            try:
                self._nvs.export_video(mid, cams, path, progress=progress)
                text = "saved %s" % osp.basename(path)
            except Exception as error:
                text = "export failed: %s" % error

            def done():
                self._nvs_status.text = text
            gui.Application.instance.post_to_main_thread(self.window, done)
        threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------------ meshes (Sub-project 2), rasterised by Open3D
    def _on_menu_open_mesh(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose a mesh", self.window.theme)
        dlg.add_filter(".obj .ply .glb .gltf", "Meshes (.obj, .ply, .glb, .gltf)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_mesh_dialog_done)
        self.window.show_dialog(dlg)

    def _on_mesh_dialog_done(self, path):
        self.window.close_dialog()
        self._mesh_list.add_item(osp.basename(path))
        self._mesh_paths.append(path)
        self._mesh_list.selected_index = len(self._mesh_paths) - 1
        self.load_mesh(path)

    def _on_mesh_selected(self, name, index):
        if 0 <= index < len(self._mesh_paths):
            self.load_mesh(self._mesh_paths[index])

    def _on_mesh_show(self, on):
        if on and not self._scene.scene.has_geometry("__mesh__") and self._mesh_paths:
            self.load_mesh(self._mesh_paths[max(0, self._mesh_list.selected_index)])
            return
        if self._scene.scene.has_geometry("__mesh__"):
            self._scene.scene.show_geometry("__mesh__", on)
        if on and self._nvs_live.checked:
            self._nvs_live.checked = False
            self._on_nvs_live(False)

    def load_mesh(self, path):
        """path: a mesh file, or an Open3D TriangleMesh (one extracted in this window)"""
        if self._scene.scene.has_geometry("__mesh__"):
            self._scene.scene.remove_geometry("__mesh__")
        if isinstance(path, o3d.geometry.TriangleMesh):
            mat = rendering.MaterialRecord()
            mat.shader = Settings.LIT
            self._scene.scene.add_geometry("__mesh__", path, mat)
            model = None
        else:
            model = o3d.io.read_triangle_model(path)
        if isinstance(path, o3d.geometry.TriangleMesh):
            pass
        elif model is not None and len(model.meshes) > 0:
            self._scene.scene.add_model("__mesh__", model)
        else:
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            mat = rendering.MaterialRecord()
            mat.shader = Settings.LIT
            self._scene.scene.add_geometry("__mesh__", mesh, mat)
        self._mesh_show.checked = True
        if self._nvs_live.checked:
            self._nvs_live.checked = False
            self._on_nvs_live(False)
        if self._nvs is None:
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())
        self.window.post_redraw()

    def load_existing_result(self, data_path):
        self._scene.scene.clear_geometry()

        self.colmap_api.data_path = data_path
        if self.colmap_api.check_colmap_folder_valid():
            self.colmap_api.estimate_cameras(recompute=False)
            self._add_geometries_from_colmap()
        else:
            self.colmap_api.data_path = None
            em = self.window.theme.font_size
            dlg = gui.Dialog("Error")

            # Add the text
            dlg_layout = gui.Vert(em / 2, gui.Margins(em, em, em, em))
            dlg_layout.add_child(gui.Label("Picked folder does not contain precomputed COLMAP data!"))

            ok = gui.Button("OK")
            ok.set_on_clicked(self._on_error_ok)

            h = gui.Horiz()
            h.add_stretch()
            h.add_child(ok)
            h.add_stretch()
            dlg_layout.add_child(h)

            dlg.add_child(dlg_layout)
            self.window.show_dialog(dlg)

    def _add_geometries_from_colmap(self):
        w = self.window
        if self.colmap_api.estimate_done():
            if self.colmap_api.estimate_error is not None:
                self.window.show_message_box(
                    "COLMAP Error",
                    str(self.colmap_api.estimate_error),
                )
                return

            self._scene.scene.add_geometry("__model__", self.colmap_api.pcd, self.settings.material)

            # Update camera list in GUI
            if not hasattr(self, '_camera_list'):
                self.colmap_ctrls.add_child(gui.Label("Camera list"))
                self._camera_list = gui.Combobox()
                self.colmap_ctrls.add_child(self._camera_list)
                self._camera_list.set_on_selection_changed(self._on_camera_list_change)

            self._camera_list.clear_items()
            for camera_name in self.colmap_api.camera_names:
                self._camera_list.add_item(camera_name)

            self._visualize_cameras()
            self._update_camera()

            w = self.window  # to make the code more concise
            w.set_needs_layout()
        else:
            gui.Application.instance.post_to_main_thread(w, self._add_geometries_from_colmap)

    def _update_camera(self):
        bounds = self._scene.scene.bounding_box
        # self._scene.setup_camera(60, bounds, bounds.get_center())

        intrinsics, extrinsics = self.colmap_api.extract_camera_parameters(self.colmap_api.activate_camera_name)
        self._scene.setup_camera(intrinsics, extrinsics, bounds)

    def _visualize_cameras(self):
        for camera_name in self.colmap_api.camera_names:
            intrinsics, extrinsics = self.colmap_api.extract_camera_parameters(camera_name)
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=intrinsics.width,
                view_height_px=intrinsics.height,
                intrinsic=intrinsics.intrinsic_matrix,
                extrinsic=extrinsics,
                scale=self.settings.camera_size
            )
            r = self.settings.camera_color.red
            g = self.settings.camera_color.green
            b = self.settings.camera_color.blue
            camera_lines.colors = o3d.utility.Vector3dVector(
                [np.array([r, g, b]),] * 8
            )
            self._scene.scene.remove_geometry(f"__cam{camera_name}__")
            self._scene.scene.add_geometry(f"__cam{camera_name}__", camera_lines, self.settings.material)

    def export_image(self, path, width, height):
        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)
