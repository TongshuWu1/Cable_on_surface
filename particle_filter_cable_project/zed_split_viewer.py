import ctypes
import sys
from pathlib import Path
from threading import Lock

import numpy as np


_FREEGLUT_HANDLE = None


def _preload_freeglut():
    if sys.platform != "win32":
        return None
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "OpenGL" / "DLLS" / "freeglut.dll",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/extras/demo_suite/freeglut.dll"),
        Path("C:/Program Files (x86)/ZED SDK/dependencies/freeglut_2.8/x64/freeglut.dll"),
    ]
    for dll_path in candidates:
        if not dll_path.exists():
            continue
        try:
            if hasattr(ctypes, "windll"):
                return ctypes.windll.LoadLibrary(str(dll_path))
            if hasattr(ctypes, "WinDLL"):
                return ctypes.WinDLL(str(dll_path))
            return ctypes.CDLL(str(dll_path))
        except OSError:
            continue
    return None


_FREEGLUT_HANDLE = _preload_freeglut()
if _FREEGLUT_HANDLE is not None:
    import OpenGL.platform

    OpenGL.platform.PLATFORM.GLUT = _FREEGLUT_HANDLE

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


_GLUT_INITIALIZED = False

UI_BG = (0.015, 0.017, 0.020)
UI_PANEL = (0.035, 0.040, 0.046)
UI_PANEL_DARK = (0.022, 0.025, 0.030)
UI_STROKE = (0.110, 0.125, 0.140)
UI_TEXT = (0.910, 0.940, 0.965)
UI_MUTED = (0.580, 0.640, 0.700)
UI_SUBTLE = (0.360, 0.410, 0.465)
UI_ACCENT = (0.270, 0.620, 0.960)
UI_ACCENT_2 = (0.250, 0.780, 0.520)
VISIBLE_COLOR = (0.25, 1.00, 0.48)
EXTENDED_COLOR = (1.00, 0.82, 0.16)
OCCLUDED_COLOR = (1.00, 0.18, 0.20)
CROSSING_ONLY_COLOR = (0.10, 0.78, 1.00)
CONTACT_COLOR = (0.20, 1.00, 0.35)
CABLE_SAMPLE_COLOR = (1.00, 0.58, 0.08)
PARTICLE_MAP_COLOR = (1.00, 1.00, 1.00)
PARTICLE_SPREAD_COLOR = (0.80, 0.52, 1.00)
START_NODE_COLOR = (0.00, 0.78, 1.00)
END_NODE_COLOR = (1.00, 0.25, 0.92)
ENDPOINT_GROUP_COLORS = (
    (1.00, 0.00, 1.00),  # endpoints_1
    (0.00, 0.86, 1.00),  # endpoints_2
    (1.00, 0.25, 0.25),  # endpoints_3
    (1.00, 0.86, 0.00),  # endpoints_4
)


def endpoint_group_color(index):
    index = max(0, int(index))
    return ENDPOINT_GROUP_COLORS[index % len(ENDPOINT_GROUP_COLORS)]

POINT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
}
"""

POINT_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 out_color;

void main() {
    out_color = vec4(v_color, 1.0);
}
"""


class ZedDepthGLViewer:
    """OpenGL UI with a 2D RGB panel and a 3D ZED point-cloud panel."""

    def __init__(
        self,
        width=1400,
        height=900,
        title="ZED Depth Point Cloud",
        window_x=40,
        window_y=40,
        left_panel_width=0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.window_x = int(window_x)
        self.window_y = int(window_y)
        self.left_panel_width = int(max(0, left_panel_width))
        self.left_panel_ratio = self._initial_left_panel_ratio()
        self.window_id = None
        self.available = False

        self.lock = Lock()
        self.pending_rgb_image = None
        self.pending_vertices = None
        self.pending_status = "waiting for frames"
        self.pending_cable_points = None
        self.pending_cable_nodes = None
        self.pending_cable_valid = None
        self.pending_cable_visible = None
        self.pending_cable_extended_visible = None
        self.pending_cable_runs = None
        self.pending_cable_runs_update = False
        self.pending_contact_observations = None
        self.pending_particle_diagnostics = None
        self.pending_coordinate_frame = None
        self.rgb_image = None
        self.vertices = np.empty((0, 6), dtype=np.float32)
        self.vertex_count = 0
        self.cable_points = np.empty((0, 3), dtype=np.float32)
        self.cable_nodes = np.empty((0, 3), dtype=np.float32)
        self.cable_valid = np.empty(0, dtype=bool)
        self.cable_visible = np.empty(0, dtype=bool)
        self.cable_extended_visible = np.empty(0, dtype=bool)
        self.cable_runs = None
        self.contact_observations = tuple()
        self.particle_diagnostics = tuple()
        self.show_particle_diagnostics = True
        self.coordinate_frame = "camera"
        self.status = "waiting for frames"

        self.vbo = None
        self.vao = None
        self.rgb_texture = None
        self.shader_program = None
        self.mvp_loc = None
        self.point_size_loc = None
        self.view_mode = "orbit"
        self.yaw_deg = -35.0
        self.pitch_deg = 22.0
        self.zoom = 1.0
        self.point_size = 2.0
        self.fov_y_deg = 70.0
        self.depth_max_m = 5.0

        self.scene_center = np.array([0.0, 0.0, -2.0], dtype=np.float32)
        self.scene_radius = 2.0
        self.has_scene = False

        self.rotating = False
        self.panning = False
        self.last_mouse = (0, 0)

    def init(self):
        global _GLUT_INITIALIZED
        if not _GLUT_INITIALIZED:
            glutInit()
            _GLUT_INITIALIZED = True

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(self.window_x, self.window_y)
        self.window_id = glutCreateWindow(self.title.encode("utf-8"))

        try:
            glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)
        except Exception:
            pass

        glDisable(GL_LIGHTING)
        self.vbo = glGenBuffers(1)
        self.rgb_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.rgb_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        try:
            self.vao = glGenVertexArrays(1)
        except Exception:
            self.vao = None
        self.shader_program = self._create_point_shader()
        if self.shader_program is not None:
            self.mvp_loc = glGetUniformLocation(self.shader_program, "u_mvp")
            self.point_size_loc = glGetUniformLocation(self.shader_program, "u_point_size")

        glViewport(0, 0, self.width, self.height)
        glClearColor(*UI_BG, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        try:
            glEnable(GL_PROGRAM_POINT_SIZE)
        except Exception:
            pass
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        glutDisplayFunc(self._draw_callback)
        glutReshapeFunc(self._reshape_callback)
        glutKeyboardFunc(self._keyboard_callback)
        glutSpecialFunc(self._special_key_callback)
        glutMouseFunc(self._mouse_callback)
        glutMotionFunc(self._motion_callback)
        try:
            glutCloseFunc(self._close_callback)
        except Exception:
            pass

        self.available = True

    def set_camera_fov(self, fov_y_deg):
        if fov_y_deg is None:
            return
        self.fov_y_deg = float(np.clip(fov_y_deg, 35.0, 100.0))

    def set_depth_max(self, depth_max_m):
        self.depth_max_m = max(1.0, float(depth_max_m))

    def is_available(self):
        return self.available

    def poll(self):
        if not self.available:
            return False
        try:
            glutSetWindow(self.window_id)
            glutPostRedisplay()
            glutMainLoopEvent()
        except Exception:
            self.available = False
        return self.available

    def close(self):
        if not self.available and self.window_id is None:
            return
        self.available = False
        try:
            if self.window_id is not None:
                try:
                    glutSetWindow(self.window_id)
                except Exception:
                    pass
            if self.shader_program is not None:
                glDeleteProgram(self.shader_program)
                self.shader_program = None
            if self.vao:
                glDeleteVertexArrays(1, [self.vao])
                self.vao = None
            if self.vbo:
                glDeleteBuffers(1, [self.vbo])
                self.vbo = None
            if self.rgb_texture:
                glDeleteTextures(1, [self.rgb_texture])
                self.rgb_texture = None
            if self.window_id is not None:
                glutDestroyWindow(self.window_id)
                self.window_id = None
        except Exception:
            pass

    def update_vertices(self, vertices, status="running"):
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 6:
            vertices = np.empty((0, 6), dtype=np.float32)
        else:
            vertices = np.ascontiguousarray(vertices, dtype=np.float32)

        center, radius = self._estimate_scene_bounds(vertices[:, :3])
        self._smooth_scene_bounds(center, radius)

        with self.lock:
            self.pending_vertices = vertices
            self.pending_status = str(status)[:96]

    def update_rgb_image(self, rgb_image):
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] < 3:
            image = None
        else:
            image = image[:, :, :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            image = np.ascontiguousarray(image)

        with self.lock:
            self.pending_rgb_image = image

    def update_cable(
        self,
        cable_points,
        cable_nodes,
        valid_nodes,
        visible_nodes=None,
        extended_visible_nodes=None,
        cable_runs=None,
        contact_observations=(),
        particle_diagnostics=(),
        coordinate_frame="camera",
    ):
        cable_points = self._as_points(cable_points)
        cable_nodes = self._as_node_points(cable_nodes)
        valid_nodes = self._as_node_mask(valid_nodes, len(cable_nodes), False)
        visible_nodes = self._as_node_mask(visible_nodes, len(cable_nodes), valid_nodes)
        extended_visible_nodes = self._as_node_mask(extended_visible_nodes, len(cable_nodes), visible_nodes)
        extended_visible_nodes = extended_visible_nodes | visible_nodes
        cable_runs = self._as_cable_runs(cable_runs, len(cable_nodes))
        particle_diagnostics = self._prepare_particle_diagnostics(particle_diagnostics)

        with self.lock:
            self.pending_cable_points = cable_points
            self.pending_cable_nodes = cable_nodes
            self.pending_cable_valid = valid_nodes
            self.pending_cable_visible = visible_nodes
            self.pending_cable_extended_visible = extended_visible_nodes
            self.pending_cable_runs = cable_runs
            self.pending_cable_runs_update = True
            self.pending_contact_observations = tuple(contact_observations or ())
            self.pending_particle_diagnostics = particle_diagnostics
            self.pending_coordinate_frame = str(coordinate_frame)

    def reset_view(self):
        self.view_mode = "orbit"
        self.yaw_deg = -35.0
        self.pitch_deg = 22.0
        self.zoom = 1.0

    def _draw_callback(self):
        if not self.available:
            return

        self._consume_pending_vertices()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        left_width, right_width = self._panel_sizes()
        if left_width > 0:
            self._draw_rgb_panel(left_width, self.height)

        glViewport(left_width, 0, right_width, self.height)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        mvp = self._set_camera(right_width, self.height)
        if self.view_mode == "orbit":
            self._draw_reference_grid()
        self._draw_point_cloud(mvp)
        self._draw_camera_origin_marker()
        self._draw_cable_overlay()
        self._draw_overlay(right_width, self.height)
        if left_width > 0:
            self._draw_divider(left_width)

        glutSwapBuffers()

    def _consume_pending_vertices(self):
        with self.lock:
            rgb_image = self.pending_rgb_image
            pending_vertices = self.pending_vertices
            pending_status = self.pending_status
            cable_points = self.pending_cable_points
            cable_nodes = self.pending_cable_nodes
            cable_valid = self.pending_cable_valid
            cable_visible = self.pending_cable_visible
            cable_extended_visible = self.pending_cable_extended_visible
            cable_runs = self.pending_cable_runs
            cable_runs_update = self.pending_cable_runs_update
            contact_observations = self.pending_contact_observations
            particle_diagnostics = self.pending_particle_diagnostics
            coordinate_frame = self.pending_coordinate_frame
            self.pending_rgb_image = None
            self.pending_vertices = None
            self.pending_cable_points = None
            self.pending_cable_nodes = None
            self.pending_cable_valid = None
            self.pending_cable_visible = None
            self.pending_cable_extended_visible = None
            self.pending_cable_runs = None
            self.pending_cable_runs_update = False
            self.pending_contact_observations = None
            self.pending_particle_diagnostics = None
            self.pending_coordinate_frame = None

        if rgb_image is not None:
            self.rgb_image = rgb_image
        if cable_points is not None:
            self.cable_points = cable_points
        if cable_nodes is not None:
            self.cable_nodes = cable_nodes
        if cable_valid is not None:
            self.cable_valid = cable_valid
        if cable_visible is not None:
            self.cable_visible = cable_visible
        if cable_extended_visible is not None:
            self.cable_extended_visible = cable_extended_visible
        if cable_runs_update:
            self.cable_runs = cable_runs
        if contact_observations is not None:
            self.contact_observations = contact_observations
        if particle_diagnostics is not None:
            self.particle_diagnostics = particle_diagnostics
        if coordinate_frame is not None:
            self.coordinate_frame = coordinate_frame

        if pending_vertices is None:
            return

        self.vertices = pending_vertices
        self.vertex_count = int(len(pending_vertices))
        self.status = pending_status

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STREAM_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _set_camera(self, viewport_width=None, viewport_height=None):
        viewport_width = self.width if viewport_width is None else max(1, int(viewport_width))
        viewport_height = self.height if viewport_height is None else max(1, int(viewport_height))
        aspect = viewport_width / viewport_height
        zfar = max(8.0, self.depth_max_m + 2.0)
        fov = float(np.clip(self.fov_y_deg, 35.0, 100.0))

        projection = self._perspective(fov, aspect, 0.05, zfar)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, aspect, 0.05, zfar)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        distance = max(0.8, self.scene_radius * 2.4) * self.zoom
        modelview = (
            self._translation(0.0, 0.0, -distance)
            @ self._rotation_x(self.pitch_deg)
            @ self._rotation_y(self.yaw_deg)
            @ self._translation(
                -float(self.scene_center[0]),
                -float(self.scene_center[1]),
                -float(self.scene_center[2]),
            )
        )
        glTranslatef(0.0, 0.0, -distance)
        glRotatef(self.pitch_deg, 1.0, 0.0, 0.0)
        glRotatef(self.yaw_deg, 0.0, 1.0, 0.0)
        glTranslatef(
            -float(self.scene_center[0]),
            -float(self.scene_center[1]),
            -float(self.scene_center[2]),
        )
        return projection @ modelview

    def _draw_rgb_panel(self, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        glViewport(0, 0, width, height)
        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        header_h = self._header_height(height)
        margin = self._panel_margin(width, height)

        self._draw_rect_2d(0, 0, width, height, UI_PANEL_DARK)
        self._draw_rect_2d(0, height - header_h, width, header_h, UI_PANEL)
        self._draw_rect_2d(0, height - header_h, 5, header_h, UI_ACCENT_2)
        self._draw_text_2d(margin, height - 27, "RGB Camera", UI_TEXT)
        if self.rgb_image is not None:
            image_h, image_w = self.rgb_image.shape[:2]
            self._draw_text_2d(
                margin,
                height - 49,
                f"{image_w} x {image_h}",
                UI_MUTED,
                GLUT_BITMAP_HELVETICA_12,
            )
        self._draw_line_2d(0, height - header_h, width, height - header_h, UI_STROKE)

        if self.rgb_image is None:
            self._draw_text_2d(margin, height - header_h - 28, "waiting for RGB frame", UI_MUTED)
            return

        image_h, image_w = self.rgb_image.shape[:2]
        if image_w <= 0 or image_h <= 0:
            return

        available_w = max(1, width - 2 * margin)
        available_h = max(1, height - header_h - 2 * margin)
        scale = min(available_w / image_w, available_h / image_h)
        draw_w = max(1, int(round(image_w * scale)))
        draw_h = max(1, int(round(image_h * scale)))
        x0 = int((width - draw_w) * 0.5)
        y0 = int((height - header_h - draw_h) * 0.5)
        x1 = x0 + draw_w
        y1 = y0 + draw_h

        glBindTexture(GL_TEXTURE_2D, self.rgb_texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            image_w,
            image_h,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            self.rgb_image,
        )

        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(float(x0), float(y0))
        glTexCoord2f(1.0, 1.0)
        glVertex2f(float(x1), float(y0))
        glTexCoord2f(1.0, 0.0)
        glVertex2f(float(x1), float(y1))
        glTexCoord2f(0.0, 0.0)
        glVertex2f(float(x0), float(y1))
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._draw_line_2d(x0, y0, x1, y0, UI_STROKE)
        self._draw_line_2d(x1, y0, x1, y1, UI_STROKE)
        self._draw_line_2d(x1, y1, x0, y1, UI_STROKE)
        self._draw_line_2d(x0, y1, x0, y0, UI_STROKE)

    def _draw_point_cloud(self, mvp):
        if self.vertex_count == 0:
            return

        if self.shader_program is not None:
            self._draw_point_cloud_shader(mvp)
            return

        self._draw_point_cloud_fixed()

    def _draw_point_cloud_shader(self, mvp):
        glPointSize(float(self.point_size))
        glUseProgram(self.shader_program)
        if self.mvp_loc is not None and self.mvp_loc >= 0:
            glUniformMatrix4fv(
                self.mvp_loc,
                1,
                GL_TRUE,
                np.ascontiguousarray(mvp, dtype=np.float32),
            )
        if self.point_size_loc is not None and self.point_size_loc >= 0:
            glUniform1f(self.point_size_loc, float(self.point_size))

        if self.vao:
            glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 6 * 4
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glDrawArrays(GL_POINTS, 0, self.vertex_count)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        if self.vao:
            glBindVertexArray(0)
        glUseProgram(0)

    def _draw_point_cloud_fixed(self):
        glPointSize(float(self.point_size))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        stride = 6 * 4
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
        glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))
        glDrawArrays(GL_POINTS, 0, self.vertex_count)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_cable_overlay(self):
        glDisable(GL_DEPTH_TEST)

        if len(self.cable_points) > 0:
            glPointSize(7.0)
            glColor3f(*CABLE_SAMPLE_COLOR)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, np.ascontiguousarray(self.cable_points, dtype=np.float32))
            glDrawArrays(GL_POINTS, 0, len(self.cable_points))
            glDisableClientState(GL_VERTEX_ARRAY)

        if self.show_particle_diagnostics:
            self._draw_particle_diagnostics()

        if self._has_cable_node_state():
            endpoint_runs = self._valid_endpoint_runs()
            endpoint_colors = self._endpoint_color_map(endpoint_runs)
            glLineWidth(8.0)
            glBegin(GL_LINES)
            for idx in range(len(self.cable_nodes) - 1):
                if not (self.cable_valid[idx] and self.cable_valid[idx + 1]):
                    continue

                p0 = self.cable_nodes[idx]
                p1 = self.cable_nodes[idx + 1]
                if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                    continue

                glColor3f(*self._segment_color(idx))
                glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
            glEnd()

            glPointSize(14.0)
            glBegin(GL_POINTS)
            for idx, point in enumerate(self.cable_nodes):
                if self.cable_valid[idx] and np.all(np.isfinite(point)):
                    glColor3f(*self._node_color(idx, endpoint_colors))
                    glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            glEnd()

            self._draw_endpoint_markers(endpoint_runs)

        self._draw_contact_observations()

        glEnable(GL_DEPTH_TEST)

    def _draw_particle_diagnostics(self):
        if not self.particle_diagnostics:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for group in self.particle_diagnostics:
            color = group["color"]
            self._draw_line_vertices(
                group["top_line_vertices"],
                color,
                alpha=0.10,
                line_width=1.0,
            )
            self._draw_line_vertices(
                group["spread_line_vertices"],
                PARTICLE_SPREAD_COLOR,
                alpha=0.90,
                line_width=2.0,
            )
            self._draw_line_vertices(
                group["map_line_vertices"],
                PARTICLE_MAP_COLOR,
                alpha=0.95,
                line_width=3.0,
            )
            self._draw_line_vertices(
                group["tangent_line_vertices"],
                UI_ACCENT_2,
                alpha=0.95,
                line_width=4.0,
            )
        glDisable(GL_BLEND)

        for group in self.particle_diagnostics:
            anchor = group["label_anchor"]
            if np.all(np.isfinite(anchor)):
                self._draw_text_3d(
                    anchor[0],
                    anchor[1],
                    anchor[2],
                    group["label"],
                    group["color"],
                    GLUT_BITMAP_HELVETICA_12,
                )

    @staticmethod
    def _draw_line_vertices(vertices, color, alpha=1.0, line_width=1.0):
        if len(vertices) == 0:
            return
        glLineWidth(float(line_width))
        glColor4f(float(color[0]), float(color[1]), float(color[2]), float(alpha))
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glDrawArrays(GL_LINES, 0, len(vertices))
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_contact_observations(self):
        for observation in self.contact_observations:
            q1 = np.asarray(getattr(observation, "q1_xyz", ()), dtype=np.float32).reshape(-1)
            q2 = np.asarray(getattr(observation, "q2_xyz", ()), dtype=np.float32).reshape(-1)
            if len(q1) < 3 or len(q2) < 3 or not np.all(np.isfinite((q1[:3], q2[:3]))):
                continue
            color = CONTACT_COLOR if bool(getattr(observation, "verified_contact", False)) else CROSSING_ONLY_COLOR
            glLineWidth(5.0)
            glColor3f(*color)
            glBegin(GL_LINES)
            glVertex3f(float(q1[0]), float(q1[1]), float(q1[2]))
            glVertex3f(float(q2[0]), float(q2[1]), float(q2[2]))
            glEnd()
            glPointSize(18.0)
            glBegin(GL_POINTS)
            glVertex3f(float(q1[0]), float(q1[1]), float(q1[2]))
            glVertex3f(float(q2[0]), float(q2[1]), float(q2[2]))
            glEnd()
            midpoint = 0.5 * (q1[:3] + q2[:3])
            calibrated = bool(getattr(observation, "diameter_calibrated", False))
            value = float(getattr(
                observation,
                "gap_m" if calibrated else "centerline_distance_m",
                np.nan,
            ))
            quantity = "gap" if calibrated else "center"
            state = "CONTACT" if bool(getattr(observation, "verified_contact", False)) else "RGB CROSSING"
            text = (
                f"{state} {quantity}={1000.0 * value:.1f}mm "
                f"s=({getattr(observation, 's1_m', np.nan):.3f},"
                f"{getattr(observation, 's2_m', np.nan):.3f})m"
            )
            self._draw_text_3d(
                midpoint[0], midpoint[1], midpoint[2], text, color, GLUT_BITMAP_HELVETICA_12
            )

    def _draw_endpoint_markers(self, endpoint_runs):
        if not endpoint_runs:
            return

        endpoints = []
        for cable_id, start_idx, end_idx in endpoint_runs:
            color = endpoint_group_color(cable_id)
            endpoints.append((start_idx, f"PF{int(cable_id) + 1} start", color, 1.0))
            endpoints.append((end_idx, f"PF{int(cable_id) + 1} end", color, -1.0))

        glPointSize(22.0)
        glBegin(GL_POINTS)
        for idx, _label, color, _direction in endpoints:
            point = self.cable_nodes[idx]
            if not (self.cable_valid[idx] and np.all(np.isfinite(point))):
                continue
            glColor3f(*color)
            glVertex3f(float(point[0]), float(point[1]), float(point[2]))
        glEnd()

        for idx, label, color, direction in endpoints:
            point = self.cable_nodes[idx]
            if not (self.cable_valid[idx] and np.all(np.isfinite(point))):
                continue
            label_pos = self._endpoint_label_position(point, direction)
            text = f"{label} {self._format_xyz(point)} m"
            self._draw_text_3d(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                text,
                color,
                GLUT_BITMAP_HELVETICA_12,
            )

    def _draw_camera_origin_marker(self):
        axis_len = float(np.clip(self.depth_max_m * 0.18, 0.12, 0.35))

        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.18, 0.16)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_len, 0.0, 0.0)
        glColor3f(0.20, 1.0, 0.35)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_len, 0.0)
        glColor3f(0.25, 0.55, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_len)
        glEnd()

        glPointSize(12.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

        self._draw_text_3d(axis_len * 1.08, 0.0, 0.0, "X", (1.0, 0.42, 0.38))
        self._draw_text_3d(0.0, axis_len * 1.08, 0.0, "Y", (0.42, 1.0, 0.55))
        self._draw_text_3d(0.0, 0.0, axis_len * 1.08, "Z", (0.45, 0.68, 1.0))
        self._draw_text_3d(0.0, -axis_len * 0.28, 0.0, "CAM 0,0,0", UI_TEXT, GLUT_BITMAP_HELVETICA_12)
        glEnable(GL_DEPTH_TEST)

    @staticmethod
    def _compile_shader(shader_type, source):
        shader_id = glCreateShader(shader_type)
        glShaderSource(shader_id, source)
        glCompileShader(shader_id)
        if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
            info = glGetShaderInfoLog(shader_id)
            glDeleteShader(shader_id)
            raise RuntimeError(info.decode("utf-8", errors="replace"))
        return shader_id

    def _create_point_shader(self):
        try:
            vertex_id = self._compile_shader(GL_VERTEX_SHADER, POINT_VERTEX_SHADER)
            fragment_id = self._compile_shader(GL_FRAGMENT_SHADER, POINT_FRAGMENT_SHADER)
            program_id = glCreateProgram()
            glAttachShader(program_id, vertex_id)
            glAttachShader(program_id, fragment_id)
            glBindAttribLocation(program_id, 0, "in_position")
            glBindAttribLocation(program_id, 1, "in_color")
            glLinkProgram(program_id)
            glDeleteShader(vertex_id)
            glDeleteShader(fragment_id)

            if glGetProgramiv(program_id, GL_LINK_STATUS) != GL_TRUE:
                info = glGetProgramInfoLog(program_id)
                glDeleteProgram(program_id)
                raise RuntimeError(info.decode("utf-8", errors="replace"))
            return program_id
        except Exception as exc:
            print(f"Point-cloud shader unavailable; using fixed-function renderer ({exc})")
            return None

    @staticmethod
    def _perspective(fov_y_deg, aspect, znear, zfar):
        f = 1.0 / np.tan(np.deg2rad(fov_y_deg) * 0.5)
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = f / aspect
        matrix[1, 1] = f
        matrix[2, 2] = (zfar + znear) / (znear - zfar)
        matrix[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
        matrix[3, 2] = -1.0
        return matrix

    @staticmethod
    def _translation(x, y, z):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = float(x)
        matrix[1, 3] = float(y)
        matrix[2, 3] = float(z)
        return matrix

    @staticmethod
    def _rotation_x(deg):
        rad = np.deg2rad(float(deg))
        c = np.cos(rad)
        s = np.sin(rad)
        matrix = np.eye(4, dtype=np.float32)
        matrix[1, 1] = c
        matrix[1, 2] = -s
        matrix[2, 1] = s
        matrix[2, 2] = c
        return matrix

    @staticmethod
    def _rotation_y(deg):
        rad = np.deg2rad(float(deg))
        c = np.cos(rad)
        s = np.sin(rad)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = c
        matrix[0, 2] = s
        matrix[2, 0] = -s
        matrix[2, 2] = c
        return matrix

    def _draw_reference_grid(self):
        center = self.scene_center
        radius = max(0.5, self.scene_radius)
        floor_y = float(center[1] - radius * 0.6)
        grid_radius = radius * 1.2
        step = max(0.1, grid_radius / 10.0)
        count = int(np.ceil(grid_radius / step))

        glDisable(GL_DEPTH_TEST)
        glLineWidth(1.0)
        glColor3f(0.16, 0.18, 0.20)
        glBegin(GL_LINES)
        for idx in range(-count, count + 1):
            offset = idx * step
            glVertex3f(float(center[0] - grid_radius), floor_y, float(center[2] + offset))
            glVertex3f(float(center[0] + grid_radius), floor_y, float(center[2] + offset))
            glVertex3f(float(center[0] + offset), floor_y, float(center[2] - grid_radius))
            glVertex3f(float(center[0] + offset), floor_y, float(center[2] + grid_radius))
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def _draw_overlay(self, width=None, height=None):
        width = self.width if width is None else max(1, int(width))
        height = self.height if height is None else max(1, int(height))
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        header_h = 82
        self._draw_rect_2d(0, height - header_h, width, header_h, (0.018, 0.021, 0.025))
        self._draw_rect_2d(0, height - header_h, 5, header_h, UI_ACCENT)
        self._draw_rect_2d(0, height - header_h, width, 1, UI_STROKE)
        self._draw_text_2d(18, height - 30, "3D Point Cloud", UI_TEXT)
        self._draw_text_2d(18, height - 54, self._compact_status(self.status, width), UI_MUTED, GLUT_BITMAP_HELVETICA_12)

        metric_y = height - 76
        metric_x = 18
        metric_x = self._draw_metric(metric_x, metric_y, "POINTS", self.vertex_count, UI_ACCENT)
        if width >= 500:
            endpoint_runs = self._valid_endpoint_runs()
            segment_count = sum(max(0, int(end_idx) - int(start_idx)) for _cable_id, start_idx, end_idx in endpoint_runs)
            cable_value = f"{segment_count} seg" if endpoint_runs else "none"
            metric_x = self._draw_metric(metric_x, metric_y, "CABLE", cable_value, UI_ACCENT_2)
        if width >= 660:
            metric_x = self._draw_metric(metric_x, metric_y, "ZOOM", f"{self.zoom:.2f}x", (0.95, 0.73, 0.24))
        if width >= 860 and self.particle_diagnostics:
            top_count = sum(int(group["top_particle_count"]) for group in self.particle_diagnostics)
            state = str(top_count) if self.show_particle_diagnostics else "off"
            self._draw_metric(metric_x, metric_y, "TOP SET", state, PARTICLE_SPREAD_COLOR)

        footer_h = 34
        self._draw_rect_2d(0, 0, width, footer_h, (0.018, 0.021, 0.025))
        self._draw_rect_2d(0, footer_h - 1, width, 1, UI_STROKE)
        footer = (
            f"Orbit: drag right panel    Zoom: wheel    Reset: R    "
            f"Particles: P ({'on' if self.show_particle_diagnostics else 'off'})    "
            f"Point size: +/- ({self.point_size:.1f})    Depth max: {self.depth_max_m:.1f}m"
        )
        self._draw_text_2d(18, 13, self._compact_status(footer, width), UI_MUTED, GLUT_BITMAP_HELVETICA_12)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def _draw_divider(self, x):
        glViewport(0, 0, self.width, self.height)
        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        self._draw_rect_2d(x - 2, 0, 4, self.height, (0.080, 0.095, 0.110))
        self._draw_rect_2d(x - 1, 0, 2, self.height, UI_STROKE)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def _draw_text_2d(self, x, y, text, color, font=GLUT_BITMAP_HELVETICA_18):
        glColor3f(float(color[0]), float(color[1]), float(color[2]))
        glRasterPos2f(float(x), float(y))
        for char in str(text):
            glutBitmapCharacter(font, ord(char))

    def _draw_text_3d(self, x, y, z, text, color, font=GLUT_BITMAP_HELVETICA_12):
        glColor3f(float(color[0]), float(color[1]), float(color[2]))
        glRasterPos3f(float(x), float(y), float(z))
        for char in str(text):
            glutBitmapCharacter(font, ord(char))

    def _draw_rect_2d(self, x, y, width, height, color):
        glColor3f(float(color[0]), float(color[1]), float(color[2]))
        x0 = float(x)
        y0 = float(y)
        x1 = float(x + width)
        y1 = float(y + height)
        glBegin(GL_QUADS)
        glVertex2f(x0, y0)
        glVertex2f(x1, y0)
        glVertex2f(x1, y1)
        glVertex2f(x0, y1)
        glEnd()

    def _draw_line_2d(self, x0, y0, x1, y1, color):
        glColor3f(float(color[0]), float(color[1]), float(color[2]))
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glVertex2f(float(x0), float(y0))
        glVertex2f(float(x1), float(y1))
        glEnd()

    def _draw_metric(self, x, y, label, value, color):
        value = str(value)
        width = max(112 if len(label) <= 5 else 134, 62 + 8 * len(value))
        height = 22
        self._draw_rect_2d(x, y, width, height, UI_PANEL)
        self._draw_rect_2d(x, y, 4, height, color)
        self._draw_text_2d(x + 10, y + 7, str(label), UI_MUTED, GLUT_BITMAP_HELVETICA_12)
        self._draw_text_2d(x + width - max(42, 8 * len(value) + 6), y + 7, value, UI_TEXT, GLUT_BITMAP_HELVETICA_12)
        return x + width + 8

    @staticmethod
    def _compact_status(status, width):
        status = str(status).replace("FUSED spatial map", "FUSED map")
        max_chars = max(42, int(width / 10))
        if len(status) <= max_chars:
            return status
        return status[: max_chars - 3] + "..."

    @staticmethod
    def _format_xyz(point):
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        if len(point) < 3 or not np.all(np.isfinite(point[:3])):
            return "(nan,nan,nan)"
        return f"({point[0]:+.3f},{point[1]:+.3f},{point[2]:+.3f})"

    def _valid_endpoint_runs(self):
        if not self._has_cable_node_state():
            return []

        finite = np.all(np.isfinite(self.cable_nodes), axis=1)
        usable = self.cable_valid & finite
        if self.cable_runs is not None:
            return [
                (int(cable_id), int(start_idx), int(end_idx))
                for cable_id, start_idx, end_idx in self.cable_runs
                if usable[int(start_idx)] and usable[int(end_idx)]
            ]
        runs = []
        start = None
        for index, is_usable in enumerate(usable):
            if is_usable and start is None:
                start = int(index)
            elif not is_usable and start is not None:
                end = int(index - 1)
                if end >= start:
                    runs.append((len(runs), start, end))
                start = None
        if start is not None:
            runs.append((len(runs), start, len(usable) - 1))
        return runs

    @staticmethod
    def _endpoint_color_map(endpoint_runs):
        colors = {}
        for cable_id, start_idx, end_idx in endpoint_runs:
            color = endpoint_group_color(cable_id)
            colors[int(start_idx)] = color
            colors[int(end_idx)] = color
        return colors

    def _endpoint_label_position(self, point, direction):
        point = np.asarray(point, dtype=np.float32).reshape(3)
        offset = float(np.clip(self.depth_max_m * 0.03, 0.035, 0.09))
        return point + np.array([direction * offset, offset, 0.0], dtype=np.float32)

    def _panel_sizes(self):
        if self.left_panel_width <= 0:
            return 0, max(1, self.width)

        min_left = min(260, max(1, self.width // 2))
        min_right = min(360, max(1, self.width // 2))
        left = int(round(self.width * self.left_panel_ratio))
        left = min(max(min_left, left), max(min_left, self.width - min_right))
        right = max(1, self.width - left)
        return left, right

    def _initial_left_panel_ratio(self):
        if self.width <= 0 or self.left_panel_width <= 0:
            return 0.0
        return float(np.clip(self.left_panel_width / max(self.width, 1), 0.25, 0.55))

    @staticmethod
    def _header_height(height):
        return int(np.clip(round(height * 0.075), 50, 68))

    @staticmethod
    def _panel_margin(width, height):
        return int(np.clip(round(min(width, height) * 0.025), 10, 18))

    def _mouse_in_cloud_panel(self, x):
        left_width, _right_width = self._panel_sizes()
        return int(x) >= left_width

    def _reshape_callback(self, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        glViewport(0, 0, self.width, self.height)

    def _keyboard_callback(self, key, _x, _y):
        if key in (b"q", b"\x1b"):
            self.close()
            return
        if key in (b"r", b"R"):
            self.reset_view()
        elif key in (b"p", b"P"):
            self.show_particle_diagnostics = not self.show_particle_diagnostics
        elif key in (b"+", b"="):
            self.point_size = min(8.0, self.point_size + 0.5)
        elif key in (b"-", b"_"):
            self.point_size = max(1.0, self.point_size - 0.5)

    def _special_key_callback(self, key, _x, _y):
        if key == GLUT_KEY_LEFT:
            self.yaw_deg -= 4.0
        elif key == GLUT_KEY_RIGHT:
            self.yaw_deg += 4.0
        elif key == GLUT_KEY_UP:
            self.pitch_deg = min(85.0, self.pitch_deg + 4.0)
        elif key == GLUT_KEY_DOWN:
            self.pitch_deg = max(-85.0, self.pitch_deg - 4.0)

    def _mouse_callback(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            self.rotating = state == GLUT_DOWN and self._mouse_in_cloud_panel(x)
            self.panning = False
            self.last_mouse = (int(x), int(y))
            return

        if button == GLUT_RIGHT_BUTTON:
            self.panning = state == GLUT_DOWN and self._mouse_in_cloud_panel(x)
            self.rotating = False
            self.last_mouse = (int(x), int(y))
            return

        if state != GLUT_DOWN:
            return

        if button == 3 and self._mouse_in_cloud_panel(x):
            self.zoom = max(0.35, self.zoom * 0.9)
        elif button == 4 and self._mouse_in_cloud_panel(x):
            self.zoom = min(3.5, self.zoom * 1.1)

    def _motion_callback(self, x, y):
        if not (self.rotating or self.panning):
            return

        last_x, last_y = self.last_mouse
        dx = int(x) - last_x
        dy = int(y) - last_y
        if self.rotating:
            self.yaw_deg += dx * 0.45
            self.pitch_deg = float(np.clip(self.pitch_deg + dy * 0.35, -85.0, 85.0))
        elif self.panning:
            scale = max(0.001, self.scene_radius * 0.0015 * self.zoom)
            yaw = np.deg2rad(self.yaw_deg)
            right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.scene_center = (self.scene_center - dx * scale * right + dy * scale * up).astype(np.float32)
        self.last_mouse = (int(x), int(y))

    def _close_callback(self):
        self.available = False

    def _smooth_scene_bounds(self, center, radius):
        if not self.has_scene:
            self.scene_center = center
            self.scene_radius = radius
            self.has_scene = True
            return

        alpha = 0.1
        self.scene_center = ((1.0 - alpha) * self.scene_center + alpha * center).astype(np.float32)
        self.scene_radius = float((1.0 - alpha) * self.scene_radius + alpha * radius)

    @staticmethod
    def _estimate_scene_bounds(points):
        if len(points) == 0:
            return np.array([0.0, 0.0, -2.0], dtype=np.float32), 2.0

        finite = points[np.all(np.isfinite(points), axis=1)]
        if len(finite) == 0:
            return np.array([0.0, 0.0, -2.0], dtype=np.float32), 2.0

        center = np.median(finite, axis=0).astype(np.float32)
        distances = np.linalg.norm(finite - center, axis=1)
        radius = float(np.percentile(distances, 95)) if len(distances) else 2.0
        return center, float(np.clip(radius, 0.5, 12.0))

    def _has_cable_node_state(self):
        count = len(self.cable_nodes)
        return (
            count > 0
            and len(self.cable_valid) == count
            and len(self.cable_visible) == count
            and len(self.cable_extended_visible) == count
        )

    def _node_color(self, idx, endpoint_colors=None):
        endpoint_colors = {} if endpoint_colors is None else endpoint_colors
        endpoint_color = endpoint_colors.get(int(idx))
        if endpoint_color is not None:
            return endpoint_color
        if self.cable_visible[idx]:
            return VISIBLE_COLOR
        if self.cable_extended_visible[idx]:
            return EXTENDED_COLOR
        return OCCLUDED_COLOR

    def _segment_color(self, idx):
        p0_visible = self.cable_visible[idx]
        p1_visible = self.cable_visible[idx + 1]
        p0_extended = self.cable_extended_visible[idx]
        p1_extended = self.cable_extended_visible[idx + 1]

        if p0_visible and p1_visible:
            return 0.12, 0.95, 0.42
        if p0_extended and p1_extended:
            return 1.0, 0.74, 0.12
        return OCCLUDED_COLOR

    @classmethod
    def _prepare_particle_diagnostics(cls, diagnostics):
        if diagnostics is None:
            return tuple()
        if hasattr(diagnostics, "top_particle_points_xyz"):
            diagnostics = ((0, diagnostics),)
        output = []
        for fallback_id, item in enumerate(diagnostics or ()):
            if isinstance(item, tuple) and len(item) == 2:
                cable_id, values = item
            else:
                cable_id, values = fallback_id, item
            try:
                cable_id = int(cable_id)
            except (TypeError, ValueError):
                cable_id = int(fallback_id)
            average = np.asarray(getattr(values, "average_points_xyz", ()), dtype=np.float32)
            map_nodes = np.asarray(getattr(values, "map_points_xyz", ()), dtype=np.float32)
            top_particles = np.asarray(getattr(values, "top_particle_points_xyz", ()), dtype=np.float32)
            principal_std = np.asarray(getattr(values, "node_principal_std_xyz", ()), dtype=np.float32)
            if (
                average.ndim != 2
                or average.shape[1] < 3
                or map_nodes.shape != average.shape
                or top_particles.ndim != 3
                or top_particles.shape[1:] != average.shape
                or principal_std.shape != average.shape
            ):
                continue
            average = np.ascontiguousarray(average[:, :3], dtype=np.float32)
            map_nodes = np.ascontiguousarray(map_nodes[:, :3], dtype=np.float32)
            top_particles = np.ascontiguousarray(top_particles[:, :, :3], dtype=np.float32)
            principal_std = np.ascontiguousarray(principal_std[:, :3], dtype=np.float32)
            if len(top_particles) == 0:
                continue

            spread_axis = 2.0 * principal_std
            spread_segments = np.stack((average - spread_axis, average + spread_axis), axis=1)
            direction_delta = np.asarray(
                getattr(values, "endpoint_direction_delta_deg", (np.nan, np.nan)),
                dtype=np.float32,
            ).reshape(-1)
            start_delta = float(direction_delta[0]) if len(direction_delta) > 0 else np.nan
            end_delta = float(direction_delta[1]) if len(direction_delta) > 1 else np.nan
            map_error = float(getattr(values, "map_to_average_node_error_m", np.nan))
            mean_spread = float(getattr(values, "mean_node_spread_m", np.nan))
            max_spread = float(getattr(values, "max_node_spread_m", np.nan))
            tangents = np.asarray(getattr(values, "endpoint_tangents_xyz", ()), dtype=np.float32)
            tangent_confidence = np.asarray(
                getattr(values, "endpoint_tangent_confidence", (np.nan, np.nan)),
                dtype=np.float32,
            ).reshape(-1)
            tangent_support = np.asarray(
                getattr(values, "endpoint_tangent_support_count", (0, 0)),
                dtype=np.int32,
            ).reshape(-1)
            tangent_segments = []
            if tangents.shape == (2, 3) and np.all(np.isfinite(tangents)):
                for endpoint_index, node_index in ((0, 0), (1, -1)):
                    confidence = float(tangent_confidence[endpoint_index]) if len(tangent_confidence) > endpoint_index else 0.0
                    length = 0.025 + 0.050 * np.clip(confidence, 0.0, 1.0)
                    tangent_segments.append((average[node_index], average[node_index] + length * tangents[endpoint_index]))
            tangent_lines = (
                np.ascontiguousarray(np.asarray(tangent_segments, dtype=np.float32).reshape(-1, 3))
                if tangent_segments
                else np.empty((0, 3), dtype=np.float32)
            )
            mean_ownership = float(getattr(values, "mean_ownership_responsibility", np.nan))
            ownership_entropy = float(getattr(values, "ownership_entropy", np.nan))
            visible_fraction = float(getattr(values, "visible_segment_fraction", np.nan))
            conditioned_ratio = float(getattr(values, "endpoint_conditioned_proposal_ratio", np.nan))
            tangent_mean = (
                float(np.mean(tangent_confidence[np.isfinite(tangent_confidence)]))
                if np.any(np.isfinite(tangent_confidence))
                else np.nan
            )
            tangent_count = int(np.sum(tangent_support)) if len(tangent_support) else 0
            label = (
                f"PF{cable_id + 1} TOP={len(top_particles)} "
                f"MAP-AVG={cls._format_mm_compact(map_error)} "
                f"spread={cls._format_mm_compact(mean_spread)}/{cls._format_mm_compact(max_spread)} "
                f"dir={cls._format_angle_compact(start_delta)}/{cls._format_angle_compact(end_delta)} "
                f"tan={cls._format_fraction_compact(tangent_mean)}/{tangent_count} "
                f"own={cls._format_fraction_compact(mean_ownership)} H={cls._format_fraction_compact(ownership_entropy)} "
                f"vis={cls._format_fraction_compact(visible_fraction)} cond={cls._format_fraction_compact(conditioned_ratio)}"
            )
            anchor = average[len(average) // 2].copy()
            anchor[1] += 0.018
            output.append({
                "cable_id": cable_id,
                "color": endpoint_group_color(cable_id),
                "top_particle_count": int(len(top_particles)),
                "top_line_vertices": cls._chain_line_vertices(top_particles),
                "map_line_vertices": cls._chain_line_vertices(map_nodes[None, :, :]),
                "spread_line_vertices": np.ascontiguousarray(spread_segments.reshape(-1, 3), dtype=np.float32),
                "tangent_line_vertices": tangent_lines,
                "label_anchor": np.ascontiguousarray(anchor, dtype=np.float32),
                "label": label,
            })
        return tuple(output)

    @staticmethod
    def _chain_line_vertices(chains):
        chains = np.asarray(chains, dtype=np.float32)
        if chains.ndim != 3 or chains.shape[1] < 2 or chains.shape[2] < 3:
            return np.empty((0, 3), dtype=np.float32)
        segments = np.stack((chains[:, :-1, :3], chains[:, 1:, :3]), axis=2).reshape(-1, 2, 3)
        segments = segments[np.all(np.isfinite(segments), axis=(1, 2))]
        return np.ascontiguousarray(segments.reshape(-1, 3), dtype=np.float32)

    @staticmethod
    def _format_mm_compact(value):
        return f"{1000.0 * float(value):.1f}mm" if np.isfinite(float(value)) else "nan"

    @staticmethod
    def _format_angle_compact(value):
        return f"{float(value):.1f}deg" if np.isfinite(float(value)) else "nan"

    @staticmethod
    def _format_fraction_compact(value):
        return f"{float(value):.2f}" if np.isfinite(float(value)) else "nan"

    @staticmethod
    def _as_points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float32)

        points = points[:, :3]
        valid = np.all(np.isfinite(points), axis=1)
        return np.ascontiguousarray(points[valid], dtype=np.float32)

    @staticmethod
    def _as_node_points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float32)

        return np.ascontiguousarray(points[:, :3], dtype=np.float32)

    @staticmethod
    def _as_node_mask(mask, node_count, default):
        node_count = int(max(0, node_count))
        if isinstance(default, np.ndarray):
            default_mask = np.asarray(default, dtype=bool).reshape(-1)
            if len(default_mask) == node_count:
                default_mask = default_mask.copy()
            else:
                default_mask = np.zeros(node_count, dtype=bool)
        else:
            default_mask = np.full(node_count, bool(default), dtype=bool)

        if mask is None:
            return default_mask

        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != node_count:
            return default_mask
        return mask.copy()

    @staticmethod
    def _as_cable_runs(runs, node_count):
        if runs is None:
            return None
        output = []
        node_count = int(max(0, node_count))
        for run in runs:
            try:
                cable_id, start_idx, end_idx = (int(value) for value in run)
            except (TypeError, ValueError):
                continue
            if cable_id < 0 or start_idx < 0 or end_idx < start_idx or end_idx >= node_count:
                continue
            output.append((cable_id, start_idx, end_idx))
        return tuple(output)
