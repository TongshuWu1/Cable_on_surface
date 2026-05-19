import ctypes
from threading import Lock

import numpy as np
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
SEG_BALL_COLOR = (1.00, 0.68, 0.10)
SEG_TABLE_COLOR = (0.22, 0.58, 1.00)
SEG_OTHER_COLOR = (1.00, 0.22, 0.72)
SPHERE_TRACK_COLORS = (
    (0.05, 1.00, 0.45),
    (1.00, 0.72, 0.10),
    (0.18, 0.74, 1.00),
    (1.00, 0.22, 0.72),
    (0.72, 0.56, 1.00),
    (0.30, 1.00, 0.82),
)
VISIBLE_COLOR = (0.25, 1.00, 0.48)
EXTENDED_COLOR = (1.00, 0.82, 0.16)
OCCLUDED_COLOR = (1.00, 0.18, 0.20)
CABLE_SAMPLE_COLOR = (1.00, 0.58, 0.08)
START_NODE_COLOR = (0.00, 0.78, 1.00)
END_NODE_COLOR = (1.00, 0.25, 0.92)

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
        self.pending_coordinate_frame = None
        self.pending_sphere_center = None
        self.pending_sphere_radius = None
        self.pending_sphere_points = None
        self.pending_spheres = None
        self.pending_segmentation_ball_points = None
        self.pending_segmentation_table_points = None
        self.pending_segmentation_other_points = None
        self.rgb_image = None
        self.vertices = np.empty((0, 6), dtype=np.float32)
        self.vertex_count = 0
        self.cable_points = np.empty((0, 3), dtype=np.float32)
        self.cable_nodes = np.empty((0, 3), dtype=np.float32)
        self.cable_valid = np.empty(0, dtype=bool)
        self.cable_visible = np.empty(0, dtype=bool)
        self.cable_extended_visible = np.empty(0, dtype=bool)
        self.sphere_center = None
        self.sphere_radius = 0.0
        self.sphere_points = np.empty((0, 3), dtype=np.float32)
        self.spheres = []
        self.segmentation_ball_points = np.empty((0, 3), dtype=np.float32)
        self.segmentation_table_points = np.empty((0, 3), dtype=np.float32)
        self.segmentation_other_points = np.empty((0, 3), dtype=np.float32)
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
        coordinate_frame="camera",
    ):
        cable_points = self._as_points(cable_points)
        cable_nodes = self._as_node_points(cable_nodes)
        valid_nodes = self._as_node_mask(valid_nodes, len(cable_nodes), False)
        visible_nodes = self._as_node_mask(visible_nodes, len(cable_nodes), valid_nodes)
        extended_visible_nodes = self._as_node_mask(extended_visible_nodes, len(cable_nodes), visible_nodes)
        extended_visible_nodes = extended_visible_nodes | visible_nodes

        with self.lock:
            self.pending_cable_points = cable_points
            self.pending_cable_nodes = cable_nodes
            self.pending_cable_valid = valid_nodes
            self.pending_cable_visible = visible_nodes
            self.pending_cable_extended_visible = extended_visible_nodes
            self.pending_coordinate_frame = str(coordinate_frame)

    def update_sphere(self, center_xyz, radius_m, surface_points=None):
        sphere = self._as_sphere_entry(center_xyz, radius_m, surface_points, track_id=1)
        with self.lock:
            self.pending_spheres = [] if sphere is None else [sphere]

    def update_spheres(self, spheres=None):
        with self.lock:
            self.pending_spheres = self._as_sphere_entries(spheres)

    def update_segmentation(self, ball_points=None, table_points=None, other_points=None):
        with self.lock:
            self.pending_segmentation_ball_points = self._as_points_or_empty(ball_points)
            self.pending_segmentation_table_points = self._as_points_or_empty(table_points)
            self.pending_segmentation_other_points = self._as_points_or_empty(other_points)

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
        self._draw_segmentation_overlay()
        self._draw_cable_overlay()
        self._draw_sphere_overlay()
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
            coordinate_frame = self.pending_coordinate_frame
            sphere_center = self.pending_sphere_center
            sphere_radius = self.pending_sphere_radius
            sphere_points = self.pending_sphere_points
            spheres = self.pending_spheres
            segmentation_ball_points = self.pending_segmentation_ball_points
            segmentation_table_points = self.pending_segmentation_table_points
            segmentation_other_points = self.pending_segmentation_other_points
            self.pending_rgb_image = None
            self.pending_vertices = None
            self.pending_cable_points = None
            self.pending_cable_nodes = None
            self.pending_cable_valid = None
            self.pending_cable_visible = None
            self.pending_cable_extended_visible = None
            self.pending_coordinate_frame = None
            self.pending_sphere_center = None
            self.pending_sphere_radius = None
            self.pending_sphere_points = None
            self.pending_spheres = None
            self.pending_segmentation_ball_points = None
            self.pending_segmentation_table_points = None
            self.pending_segmentation_other_points = None

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
        if coordinate_frame is not None:
            self.coordinate_frame = coordinate_frame
        if sphere_center is not None:
            self.sphere_center = sphere_center
        if sphere_radius is not None:
            self.sphere_radius = float(max(0.0, sphere_radius))
        if sphere_points is not None:
            self.sphere_points = sphere_points
        if spheres is not None:
            self.spheres = spheres
            if len(spheres) > 0:
                self.sphere_center = spheres[0]["center"]
                self.sphere_radius = spheres[0]["radius"]
                self.sphere_points = spheres[0]["points"]
            else:
                self.sphere_center = None
                self.sphere_radius = 0.0
                self.sphere_points = np.empty((0, 3), dtype=np.float32)
        if segmentation_ball_points is not None:
            self.segmentation_ball_points = segmentation_ball_points
        if segmentation_table_points is not None:
            self.segmentation_table_points = segmentation_table_points
        if segmentation_other_points is not None:
            self.segmentation_other_points = segmentation_other_points

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
            glBegin(GL_POINTS)
            for point in self.cable_points:
                glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            glEnd()

        if self._has_cable_node_state():
            start_idx, end_idx = self._valid_endpoint_indices()
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
                    glColor3f(*self._node_color(idx, start_idx, end_idx))
                    glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            glEnd()

            self._draw_endpoint_markers(start_idx, end_idx)

        glEnable(GL_DEPTH_TEST)

    def _draw_segmentation_overlay(self):
        if (
            len(self.segmentation_ball_points) == 0
            and len(self.segmentation_table_points) == 0
            and len(self.segmentation_other_points) == 0
        ):
            return

        glDisable(GL_DEPTH_TEST)
        self._draw_point_set(self.segmentation_table_points, SEG_TABLE_COLOR, point_size=4.5, max_points=900)
        self._draw_point_set(self.segmentation_other_points, SEG_OTHER_COLOR, point_size=4.5, max_points=700)
        self._draw_point_set(self.segmentation_ball_points, SEG_BALL_COLOR, point_size=7.0, max_points=900)
        glEnable(GL_DEPTH_TEST)

    def _draw_point_set(self, points, color, point_size=5.0, max_points=900):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return

        max_points = max(1, int(max_points))
        step = max(1, len(points) // max_points)
        glPointSize(float(point_size))
        glColor3f(float(color[0]), float(color[1]), float(color[2]))
        glBegin(GL_POINTS)
        for point in points[::step]:
            glVertex3f(float(point[0]), float(point[1]), float(point[2]))
        glEnd()

    def _draw_sphere_overlay(self):
        spheres = self.spheres
        if not spheres and self.sphere_center is not None:
            spheres = [
                {
                    "center": self.sphere_center,
                    "radius": self.sphere_radius,
                    "points": self.sphere_points,
                    "track_id": 1,
                }
            ]
        if not spheres:
            return

        glDisable(GL_DEPTH_TEST)
        for index, sphere in enumerate(spheres):
            center = sphere["center"]
            radius = float(sphere["radius"])
            points = sphere["points"]
            color = SPHERE_TRACK_COLORS[index % len(SPHERE_TRACK_COLORS)]
            if len(center) < 3 or radius <= 0.0 or not np.all(np.isfinite(center)):
                continue

            if len(points) > 0:
                step = max(1, len(points) // 350)
                glPointSize(5.5)
                glColor3f(*color)
                glBegin(GL_POINTS)
                for point in points[::step]:
                    glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                glEnd()

            glPushMatrix()
            glTranslatef(float(center[0]), float(center[1]), float(center[2]))
            try:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(float(color[0]), float(color[1]), float(color[2]), 0.18)
                glutSolidSphere(radius, 20, 10)
                glDisable(GL_BLEND)
            except Exception:
                pass
            glColor3f(*color)
            glLineWidth(2.0)
            glutWireSphere(radius, 20, 10)
            glPopMatrix()

            glPointSize(11.0)
            glColor3f(*color)
            glBegin(GL_POINTS)
            glVertex3f(float(center[0]), float(center[1]), float(center[2]))
            glEnd()
        glEnable(GL_DEPTH_TEST)

    def _draw_endpoint_markers(self, start_idx, end_idx):
        if start_idx is None or end_idx is None:
            return

        endpoints = [
            (start_idx, "START", START_NODE_COLOR, 1.0),
            (end_idx, "END", END_NODE_COLOR, -1.0),
        ]

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
            print(f"Point-cloud shader unavailable; using fixed-function fallback ({exc})")
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
            sphere_value = f"{self.sphere_radius:.3f}m" if self.sphere_center is not None else "none"
            metric_x = self._draw_metric(metric_x, metric_y, "SPHERE", sphere_value, UI_ACCENT_2)
        if width >= 660:
            self._draw_metric(metric_x, metric_y, "ZOOM", f"{self.zoom:.2f}x", (0.95, 0.73, 0.24))

        footer_h = 34
        self._draw_rect_2d(0, 0, width, footer_h, (0.018, 0.021, 0.025))
        self._draw_rect_2d(0, footer_h - 1, width, 1, UI_STROKE)
        footer = (
            f"Orbit: drag right panel    Zoom: wheel    Reset: R    "
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
        status = str(status).replace("LIVE depth fallback", "LIVE depth").replace("FUSED spatial map", "FUSED map")
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

    def _valid_endpoint_indices(self):
        if not self._has_cable_node_state():
            return None, None

        finite = np.all(np.isfinite(self.cable_nodes), axis=1)
        usable = self.cable_valid & finite
        indices = np.flatnonzero(usable)
        if len(indices) == 0:
            return None, None

        return int(indices[0]), int(indices[-1])

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
        if key == b"r":
            self.reset_view()
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

    def _node_color(self, idx, start_idx=None, end_idx=None):
        if start_idx is not None and int(idx) == int(start_idx):
            return START_NODE_COLOR
        if end_idx is not None and int(idx) == int(end_idx):
            return END_NODE_COLOR
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

    @staticmethod
    def _as_points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float32)

        points = points[:, :3]
        valid = np.all(np.isfinite(points), axis=1)
        return np.ascontiguousarray(points[valid], dtype=np.float32)

    @classmethod
    def _as_points_or_empty(cls, points):
        if points is None:
            return np.empty((0, 3), dtype=np.float32)
        return cls._as_points(points)

    @classmethod
    def _as_sphere_entries(cls, spheres):
        if spheres is None:
            return []
        entries = []
        for index, sphere in enumerate(spheres):
            if isinstance(sphere, dict):
                center = sphere.get("center")
                radius = sphere.get("radius", 0.0)
                points = sphere.get("points", None)
                track_id = sphere.get("track_id", index + 1)
            else:
                try:
                    center, radius, points = sphere[:3]
                except Exception:
                    continue
                track_id = index + 1
            entry = cls._as_sphere_entry(center, radius, points, track_id=track_id)
            if entry is not None:
                entries.append(entry)
        return entries

    @classmethod
    def _as_sphere_entry(cls, center_xyz, radius_m, surface_points=None, track_id=0):
        if center_xyz is None:
            return None
        center = np.asarray(center_xyz, dtype=np.float32).reshape(-1)
        if len(center) < 3 or not np.all(np.isfinite(center[:3])):
            return None
        try:
            radius = float(radius_m)
        except Exception:
            return None
        if not np.isfinite(radius) or radius <= 0.0:
            return None
        return {
            "center": np.ascontiguousarray(center[:3], dtype=np.float32),
            "radius": float(radius),
            "points": cls._as_points_or_empty(surface_points),
            "track_id": int(track_id),
        }

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
            fallback = np.asarray(default, dtype=bool).reshape(-1)
            if len(fallback) == node_count:
                fallback = fallback.copy()
            else:
                fallback = np.zeros(node_count, dtype=bool)
        else:
            fallback = np.full(node_count, bool(default), dtype=bool)

        if mask is None:
            return fallback

        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != node_count:
            return fallback
        return mask.copy()
