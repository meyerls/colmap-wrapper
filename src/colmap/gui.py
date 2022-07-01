#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import sys
import platform

# Libs
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Own modules
import colmap
import utils
import image

isMacOS = (platform.system() == "Darwin")


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.show_cameras = False
        self.show_images = False

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_COLMAP = 4
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.DOUBLE)
        self._point_size.set_limits(0.1, 5)
        self._point_size.double_value = 1.5
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        # grid.add_child(gui.Label("Material"))
        # grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point radius"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        colmap_settings = gui.CollapsableVert("COLMAP settings", 0,
                                              gui.Margins(em, 0, 0, 0))

        self._show_cameras = gui.Checkbox("Show Cameras")
        self._show_cameras.set_on_checked(self._on_show_cameras)
        self._show_cameras.visible = False
        self._resize_cameras = gui.Slider(gui.Slider.DOUBLE)
        self._resize_cameras.set_limits(0., 1.)
        self._resize_cameras.double_value = .2
        self._resize_cameras.visible = False
        self._resize_cameras.set_on_value_changed(self._on_resize_camera)
        self._show_images = gui.Checkbox("Show Images")
        self._show_images.set_on_checked(self._on_show_images)
        self._show_images.visible = False

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Cameras"))
        grid.add_child(self._show_cameras)
        grid.add_child(gui.Label("Camera Size"))
        grid.add_child(self._resize_cameras)
        grid.add_child(gui.Label("Images"))
        grid.add_child(self._show_images)
        colmap_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(colmap_settings)
        # ----

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
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Load COLMAP...", AppWindow.MENU_COLMAP)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
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
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_COLMAP,
                                     self._on_menu_colmap)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._show_cameras.checked = self.settings.show_cameras
        self._show_images.checked = self.settings.show_images

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._material_prefab.enabled = (
                self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + radius) of every
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

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_menu_colmap(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose COLMAp folder to load",
                             self.window.theme)
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_colmap_dialog_done)
        self.window.show_dialog(dlg)

        self._show_cameras.visible = True

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_show_cameras(self, show):
        self.settings.show_cameras = show

        self._show_images.visible = show
        self._resize_cameras.visible = show

        if self.settings.show_cameras:
            self.camera_line_set = []

            for idx in range(1, self.colmap_project.images.__len__() + 1):
                camera_intrinsics = image.Intrinsics(self.colmap_project.cameras[1])
                camera_intrinsics.load_from_colmap(
                    camera=self.colmap_project.cameras[self.colmap_project.images[idx].camera_id])

                Rwc, twc, M = utils.convert_colmap_extrinsics(frame=self.colmap_project.images[idx])

                line_set, sphere, mesh = draw_camera_viewport(extrinsics=M, intrinsics=camera_intrinsics)

                self._scene.scene.add_geometry("Line_set_{}".format(idx), line_set,
                                               self.settings.material)
                self.camera_line_set.append("Line_set_{}".format(idx))
                # self._scene.scene.add_geometry("Sphere_set_{}".format(idx), sphere,
                #                               self.settings.material)
                # self.camera_geometries.append("Sphere_set_{}".format(idx))
                # self._scene.scene.add_geometry("Mesh_set_{}".format(idx), mesh,
                #                               self.settings.material)
                # self.camera_geometries.append("Mesh_set_{}".format(idx))

        else:
            for geometry_name in self.camera_line_set:
                self._scene.scene.remove_geometry(geometry_name)

        self._apply_settings()

    def _on_resize_camera(self, size):

        for geometry_name in self.camera_line_set:
            self._scene.scene.remove_geometry(geometry_name)

        self.camera_line_set = []

        for idx in range(1, self.colmap_project.images.__len__() + 1):
            camera_intrinsics = image.Intrinsics(self.colmap_project.cameras[1])
            camera_intrinsics.load_from_colmap(
                camera=self.colmap_project.cameras[self.colmap_project.images[idx].camera_id])
            camera_intrinsics.cx = 3000
            camera_intrinsics.cy = 2000

            Rwc, twc, M = utils.convert_colmap_extrinsics(frame=self.colmap_project.images[idx])

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=M, intrinsics=camera_intrinsics, scale=size)

            self._scene.scene.add_geometry("Line_set_{}".format(idx), line_set,
                                           self.settings.material)
            self.camera_line_set.append("Line_set_{}".format(idx))
            # self._scene.scene.add_geometry("Sphere_set_{}".format(idx), sphere,
            #                               self.settings.material)
            # self.camera_geometries.append("Sphere_set_{}".format(idx))
            # self._scene.scene.add_geometry("Mesh_set_{}".format(idx), mesh,
            #                               self.settings.material)
            # self.camera_geometries.append("Mesh_set_{}".format(idx))

    def _on_show_images(self, show):
        self.settings.show_images = show

        if self.settings.show_images:
            self.camera_mesh = []

            for idx in range(1, self.colmap_project.images.__len__() + 1):
                camera_intrinsics = image.Intrinsics(self.colmap_project.cameras[1])
                camera_intrinsics.load_from_colmap(
                    camera=self.colmap_project.cameras[self.colmap_project.images[idx].camera_id])
                camera_intrinsics.cx = 3000
                camera_intrinsics.cy = 2000

                Rwc, twc, M = utils.convert_colmap_extrinsics(frame=self.colmap_project.images[idx])

                image_path = "C:\\Users\\se86kimy\\Dropbox\\05_productive\\01_code\\02_Reconstruction\\03_scale_ambiguity\\data\\scenario_05\\orig\\{}".format(
                    self.colmap_project.images[idx].name)
                _, _, mesh = draw_camera_viewport(extrinsics=M, intrinsics=camera_intrinsics, image=image_path,
                                                  scale=self._resize_cameras.double_value)

                # self._scene.scene.add_geometry("Line_set_{}".format(idx), line_set,
                #                               self.settings.material)
                # self.camera_line_set.append("Line_set_{}".format(idx))
                # self._scene.scene.add_geometry("Sphere_set_{}".format(idx), sphere,
                #                               self.settings.material)
                # self.camera_geometries.append("Sphere_set_{}".format(idx))
                self._scene.scene.add_geometry("Mesh_set_{}".format(idx), mesh,
                                               self.settings.material)
                self.camera_mesh.append("Mesh_set_{}".format(idx))

        else:
            for geometry_name in self.camera_mesh:
                self._scene.scene.remove_geometry(geometry_name)

        self._apply_settings()

        self._apply_settings()

    def _on_load_colmap_dialog_done(self, foldername):
        self.window.close_dialog()
        self.colmap_project = colmap.COLMAP(project_path=foldername)

        geometry = None

        if geometry is None:
            cloud = None
            try:
                cloud = self.colmap_project.get_dense()
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read PCD")
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points")

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               rendering.MaterialRecord())
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

        self._apply_settings()

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
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the radius
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum radius.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def draw_image2camera_mesh(extrinsics, intrinsics, image, scale=1):
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]

    line_set, points = draw_camera_plane(extrinsics, intrinsics, scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))

    normal_vec = - (np.asarray([0, 0, intrinsics.fx]) @ R.T)
    pcd.normals = o3d.utility.Vector3dVector(np.tile(normal_vec, (pcd.points.__len__(), 1)))

    # ToDo:
    # plane = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #    pcd,
    #    o3d.utility.DoubleVector([2, 1 * 2]))
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = pcd.points
    plane.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 3],
                                                             [1, 2, 3]]))
    plane.compute_vertex_normals()

    # o3d.visualization.draw_geometries([line_set, plane])

    text = cv2.imread(image)
    text = cv2.cvtColor(text, cv2.COLOR_BGR2RGB)

    v_uv = np.asarray([
        [1, 1],
        [1, 0],
        [0, 1],

        [1, 0],
        [0, 0],
        [0, 1],
    ])

    plane.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
    plane.triangle_material_ids = o3d.utility.IntVector([0] * 2)
    plane.textures = [o3d.geometry.Image(text)]

    return plane


def draw_camera_viewport(extrinsics, intrinsics, image=None, scale=1):
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]

    max_norm = max(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    normal_vec = (np.asarray([0, 0, 10]) @ R.T)

    points = [
        t,
        t + normal_vec,
        t + (np.asarray([intrinsics.cx, intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([intrinsics.cx, -intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-intrinsics.cx, -intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-intrinsics.cx, intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
    ]

    lines = [
        # [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 2],
    ]

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pc, line_set])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(t)

    if image != None:
        mesh = draw_image2camera_mesh(extrinsics=extrinsics, intrinsics=intrinsics, image=image, scale=scale)
    else:
        mesh = o3d.geometry.TriangleMesh()

    return line_set, sphere, mesh


def draw_camera_plane(extrinsics, intrinsics, scale):
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]

    max_norm = max(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

    points_camera_plane = [
        t + (np.asarray([intrinsics.cx, intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([intrinsics.cx, -intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-intrinsics.cx, -intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
        t + (np.asarray([-intrinsics.cx, intrinsics.cy, intrinsics.fx]) * scale) / max_norm @ R.T,
    ]

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ]

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_camera_plane),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set, points_camera_plane


def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024*3, 768*3)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
