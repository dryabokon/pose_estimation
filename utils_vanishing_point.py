#https://github.com/rayryeng/XiaohuLuVPDetection/blob/master/lu_vp_detect/run_vp_detect.py
#https://github.com/AngeloG98/VanishingPointCameraCalibration
#https://github.com/chsasank/Image-Rectification/blob/master/rectification.py
import math
import pandas as pd
import numpy
import cv2
import inspect
# ---------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_IO
import tools_image
import tools_plot_v2
import tools_render_CV
import tools_time_profiler
import tools_DF
from CV import tools_alg_match
from CV import tools_Skeletone
# ---------------------------------------------------------------------------------------------------------------------
class detector_VP:
    def __init__(self, folder_out):
        self.folder_out = folder_out
        self.Ske = tools_Skeletone.Skelenonizer(folder_out=folder_out)
        self.P = tools_plot_v2.Plotter(folder_out=folder_out)

        self.TP = tools_time_profiler.Time_Profiler()

        self.detector = 'ORB'
        self.matchtype = 'knn'
        self.kernel_conv_vert = (7,3)
        self.kernel_conv_horz = (3,7)
        self.H = 1080
        self.W = 1920
        self.ratio_target = 5.0 / 1.6

        self.config_algo_ver_lines = 'LSD'
        self.tol_deg_hor_line = 20
        self.color_markup_grid = (0,0,0)
        self.width_markup_grid = 2
        self.lines_width = 2
        self.color_markup_cuboid = tools_draw_numpy.color_black
        self.transp_markup = 0.85
        self.colors_rag = tools_draw_numpy.get_colors(255,colormap = 'nipy_spectral')[120:240]
        self.font_size = 28

        return
# ---------------------------------------------------------------------------------------------------------------------
    def iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
# ---------------------------------------------------------------------------------------------------------------------
    def keep_lines_by_length(self, lines, len_min=15, len_max=200,inv=False):
        nrm = numpy.array([numpy.linalg.norm(l[:2] - l[2:]) for l in lines])
        idx = (nrm >= len_min) & (nrm <= len_max)
        if inv:
            idx=~idx
        lines_res = numpy.array(lines)[idx]
        return lines_res
# ---------------------------------------------------------------------------------------------------------------------
    def keep_lines_by_angle(self, lines, angle_deg_min, angle_deg_max,inv=False):
        angles = numpy.array([self.get_angle_deg(line) for line in lines])
        idx = (angles >= angle_deg_min) & (angles <= angle_deg_max)
        if inv:
            idx=~idx
        lines_res = numpy.array(lines)[idx]
        return lines_res
# ---------------------------------------------------------------------------------------------------------------------
    def keep_lines_above_cutoff_line(self, lines, line_cutoff, inv=False):
        if lines is None or len(lines)==0: return lines
        idx = numpy.array([tools_render_CV.is_point_above_line(line[:2],line_cutoff) and tools_render_CV.is_point_above_line(line[2:],line_cutoff) for line in lines])
        if inv:
            idx=~idx
        lines_res = numpy.array(lines)[idx]
        return lines_res
# ---------------------------------------------------------------------------------------------------------------------
    def save_lines(self,lines,filename_out):
        pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2']).to_csv(self.folder_out + filename_out, index=False)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def load_lines(self,filename_in):
        lines = pd.read_csv(filename_in).values
        return lines
# ---------------------------------------------------------------------------------------------------------------------
    def reshape_lines_as_paired(self,lines):
        res = numpy.array([[[l[0],l[1]],[l[2],l[3]]] for l in lines])
        return res
# ---------------------------------------------------------------------------------------------------------------------
    def reshape_lines_as_flat(self, lines):
        res = numpy.array([[l[0][0],l[0][1],l[1][0],l[1][1]] for l in lines])
        return res
# ---------------------------------------------------------------------------------------------------------------------
    def get_angle_deg(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            angle = 0
        else:
            angle = 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
        return angle
# ----------------------------------------------------------------------------------------------------------------------
    def boxify_lines(self, lines, box, do_quick_unstable=False, do_debug=False):
        # visible lines are extended into the boarder of the box

        def is_inside_line(p, p1, p2):
            a = numpy.linalg.norm(p1 - p)
            b = numpy.linalg.norm(p2 - p)
            c = numpy.linalg.norm(p1 - p2)
            res = a + b <= c + 0.1
            return res

        def is_inside_box(p, left, top, right, bottom):
            res = (left <= p[0] <= right) and (top <= p[1] <= bottom)
            return res

        # W,H = box[2], box[3]
        tol = 2
        results, idx = [], []
        left, top, right, bottom = box[0], box[1], box[2], box[3]

        segments = [(left, top, right, top), (right, top, right, bottom), (left, bottom, right, bottom),
                    (left, top, left, bottom)]

        for l, line in enumerate(lines):
            if numpy.any(numpy.isnan(line)): continue
            if numpy.linalg.norm(line) == 0: continue
            result = []

            is_in_box = is_inside_box((line[0], line[1]), box[0], box[1], box[2], box[3]) or is_inside_box(
                (line[2], line[3]), box[0], box[1], box[2], box[3])
            if do_quick_unstable:
                x1, y1 = tools_render_CV.line_intersection_unstable(line, segments[0])
                x2, y2 = tools_render_CV.line_intersection_unstable(line, segments[1])
                x3, y3 = tools_render_CV.line_intersection_unstable(line, segments[2])
                x4, y4 = tools_render_CV.line_intersection_unstable(line, segments[3])
            else:
                x1, y1 = tools_render_CV.line_intersection(line, segments[0])
                x2, y2 = tools_render_CV.line_intersection(line, segments[1])
                x3, y3 = tools_render_CV.line_intersection(line, segments[2])
                x4, y4 = tools_render_CV.line_intersection(line, segments[3])

            if x1 is not numpy.nan and y1 is not numpy.nan:
                if left <= x1 + tol and x1 - tol <= right and abs(y1 - top) <= tol:
                    if is_in_box or is_inside_line((x1, y1), line[:2], line[2:]):
                        result.append((x1, top))

            if x2 is not numpy.nan and y2 is not numpy.nan:
                if top <= y2 + tol and y2 - tol <= bottom and abs(x2 - right) <= tol:
                    if is_in_box or is_inside_line((x2, y2), line[:2], line[2:]):
                        result.append((right, y2))

            if x3 is not numpy.nan and y3 is not numpy.nan:
                if left <= x3 + tol and x3 - tol <= right and abs(y3 - bottom) <= tol:
                    if is_in_box or is_inside_line((x3, y3), line[:2], line[2:]):
                        result.append((x3, bottom))

            if x4 is not numpy.nan and y4 is not numpy.nan:
                if top <= y4 + tol and y4 - tol <= bottom and abs(x4 - left) <= tol:
                    if is_in_box or is_inside_line((x4, y4), line[:2], line[2:]):
                        result.append((left, y4))

            if len(result) >= 2:
                results.append((result[0][0], result[0][1], result[1][0], result[1][1]))
                idx.append(l)

            if do_debug:
                image = numpy.full((bottom, right, 3), 64, dtype=numpy.uint8)
                box_p1 = tools_draw_numpy.extend_view((left, top), bottom, right, factor=4)
                box_p2 = tools_draw_numpy.extend_view((right, bottom), bottom, right, factor=4)
                line_p1 = tools_draw_numpy.extend_view((line[0], line[1]), bottom, right, factor=4)
                line_p2 = tools_draw_numpy.extend_view((line[2], line[3]), bottom, right, factor=4)

                circle_p1 = tools_draw_numpy.extend_view((x1, y1), bottom, right, factor=4)
                circle_p2 = tools_draw_numpy.extend_view((x2, y2), bottom, right, factor=4)
                circle_p3 = tools_draw_numpy.extend_view((x3, y3), bottom, right, factor=4)
                circle_p4 = tools_draw_numpy.extend_view((x4, y4), bottom, right, factor=4)

                cv2.rectangle(image, tuple(box_p1), tuple(box_p2), tools_draw_numpy.color_blue, thickness=2)
                cv2.line(image, tuple(line_p1), tuple(line_p2), tools_draw_numpy.color_red, thickness=4)

                if len(result) >= 2:
                    res_p1 = tools_draw_numpy.extend_view((result[0][0], result[0][1]), bottom, right, factor=4)
                    res_p2 = tools_draw_numpy.extend_view((result[1][0], result[1][1]), bottom, right, factor=4)
                    cv2.line(image, tuple(res_p1), tuple(res_p2), tools_draw_numpy.color_amber, thickness=1)

                cv2.circle(image, (circle_p1[0], circle_p1[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)
                cv2.circle(image, (circle_p2[0], circle_p2[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)
                cv2.circle(image, (circle_p3[0], circle_p3[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)
                cv2.circle(image, (circle_p4[0], circle_p4[1]), 10, tools_draw_numpy.color_light_gray, thickness=2)

                cv2.imwrite(self.folder_out + 'boxify.png', image)
                uu = 0

        return numpy.array(results, dtype=numpy.int)
# -----------------------------------------------------------------------
    def get_lines_ver_candidates_static(self,folder_in,len_min=15, len_max=200,do_debug=False):

        filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        lines = None

        for filename in filenames[:10]:
            image = cv2.imread(folder_in + filename)
            if self.config_algo_ver_lines=='LSD':
                lns = self.Ske.detect_lines_LSD(image)
            else:
                img_amp = self.Ske.preprocess_amplify(image, self.kernel_conv_vert)
                img_bin = self.Ske.binarize(img_amp)
                img_ske = cv2.Canny(image=img_bin, threshold1=20, threshold2=250)
                segments = self.Ske.skeleton_to_segments(img_ske)
                segments_straight = self.Ske.sraighten_segments(segments, min_len=10)
                segments_long = self.Ske.filter_short_segments2(segments_straight, ratio=0.10)
                lns = self.Ske.interpolate_segments_by_lines(segments_long)
            lines = lns if lines is None else numpy.concatenate([lines,lns],axis=0)

        lines = self.keep_lines_by_length(lines, len_min, len_max)
        lines = self.keep_lines_by_angle(lines, 90  - self.tol_deg_hor_line, 90  + self.tol_deg_hor_line, inv=True)
        lines = self.keep_lines_by_angle(lines, 270 - self.tol_deg_hor_line, 270 + self.tol_deg_hor_line, inv=True)
        lines = self.keep_lines_by_angle(lines, 0, 0, inv=True)
        lines = self.keep_lines_by_angle(lines, 180, 180, inv=True)

        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_static_ver.png',tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, color=(0, 0, 200), w=1))

        return lines
# -----------------------------------------------------------------------

    def get_lines_ver_candidates_dynamic(self,folder_in,len_min=15,len_max=200,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        filenames = tools_IO.get_filenames(folder_in,'*.jpg')
        img_cur = cv2.imread(folder_in + filenames[0])

        lines = []
        for filename_cur in filenames[1:200]:
            img_prev = img_cur.copy()
            img_cur  = cv2.imread(folder_in+filename_cur)

            points1, des1 = tools_alg_match.get_keypoints_desc(img_cur, self.detector)
            points2, des2 = tools_alg_match.get_keypoints_desc(img_prev, self.detector)
            match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2,self.matchtype)
            for m1, m2 in zip(match1, match2):
                if numpy.linalg.norm(m1-m2)<len_min:continue
                if numpy.linalg.norm(m1-m2)>len_max:continue
                lines.append([m1[0], m1[1], m2[0], m2[1]])

        lines = numpy.array(lines).reshape((-1,4))
        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_dynamic_ver.png',
                        tools_draw_numpy.draw_lines(tools_image.desaturate(img_cur), lines, color=(0, 0, 200), w=1))
        self.save_lines(lines,'lines_ver_dyn.csv')
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return lines

# ----------------------------------------------------------------------------------------------------------------------
    def get_lines_hor_candidates_static(self, folder_in, len_min=15,len_max=200,do_debug=False):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        lines = None

        tol_h = 20

        for filename in filenames[:100]:
            image = cv2.imread(folder_in + filename)
            # image_skeleton = cv2.Canny(image=image, threshold1=20, threshold2=250)
            # segments = self.Ske.skeleton_to_segments(image_skeleton)
            # segments_straight = self.Ske.sraighten_segments(segments, min_len=20)
            # segments_long = self.Ske.filter_short_segments2(segments_straight, ratio=0.10)
            # lns = self.Ske.interpolate_segments_by_lines(segments_long)
            lns = self.Ske.detect_lines_LSD(image)
            lines = lns if lines is None else numpy.concatenate([lines, lns], axis=0)

        H,W = image.shape[:2]
        lines = self.keep_lines_by_length(lines, len_min, len_max)
        lines = numpy.concatenate([self.keep_lines_by_angle(lines, 270-self.tol_deg_hor_line, 270 + self.tol_deg_hor_line),self.keep_lines_by_angle(lines, 90-self.tol_deg_hor_line, 90+self.tol_deg_hor_line)])
        lines = self.keep_lines_above_cutoff_line(lines, (0,H-tol_h,W,H-tol_h))

        if do_debug:
            cv2.imwrite(self.folder_out + 'lines_static_hor.png',
                        tools_draw_numpy.draw_lines(tools_image.desaturate(image), lines, color=(0, 0, 200), w=1))
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return lines
 # -----------------------------------------------------------------------
    def get_focal_length(self, vps):
        #Focal length of the camera in pixels
        pp = [self.W / 2, self.H / 2]
        if vps[0][0] == vps[1][0]:return math.fabs(pp[0] - vps[0][0])
        if vps[0][1] == vps[1][1]:return math.fabs(pp[1] - vps[0][1])
        k_uv = (vps[0][1] - vps[1][1]) / (vps[0][0] - vps[1][0])
        b_uv = vps[1][1] - k_uv * vps[1][0]
        pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(k_uv * k_uv + 1, 0.5)
        lenth_uv = math.sqrt((vps[0][1] - vps[1][1]) ** 2 + (vps[0][0] - vps[1][0]) ** 2)
        lenth_pu = math.sqrt((vps[0][1] - pp[1]) ** 2 + (vps[0][0] - pp[0]) ** 2)
        up_uv = math.sqrt(lenth_pu ** 2 - pp_uv ** 2)
        vp_uv = abs(lenth_uv - up_uv)
        focal_length = math.sqrt((up_uv * vp_uv) - ((pp_uv) ** 2))
        return focal_length
# -----------------------------------------------------------------------
    def calculate_metric_angle(self, current_hypothesis, lines, ignore_pts, ransac_angle_thresh):
        current_hypothesis = current_hypothesis / current_hypothesis[-1]
        hypothesis_vp_direction = current_hypothesis[:2] - lines[:, 0]
        lines_vp_direction = lines[:, 1] - lines[:, 0]
        magnitude = numpy.linalg.norm(hypothesis_vp_direction, axis=1) * numpy.linalg.norm(lines_vp_direction, axis=1)
        magnitude[magnitude == 0] = 1e-5
        cos_theta = (hypothesis_vp_direction * lines_vp_direction).sum(axis=-1) / magnitude
        theta = numpy.arccos(numpy.abs(numpy.clip(cos_theta,-1,1)))
        inliers = (theta < ransac_angle_thresh * numpy.pi / 180)
        inliers[ignore_pts] = False
        return inliers, inliers.sum()
# ----------------------------------------------------------------------------------------------------------------------
    def run_line_ransac(self,lines, ransac_iter=3000, ransac_angle_thresh=2.0, ignore_pts=None):
        best_vote_count = 0
        idx_best_inliers = None
        best_hypothesis = None
        if ignore_pts is None:
            ignore_pts = numpy.zeros((lines.shape[0])).astype('bool')
            lines_to_chose = numpy.arange(lines.shape[0])
        else:
            lines_to_chose = numpy.where(ignore_pts == 0)[0]
        for iter_count in range(ransac_iter):
            idx1, idx2 = numpy.random.choice(lines_to_chose, 2, replace=False)
            l1 = numpy.cross(numpy.append(lines[idx1][1], 1), numpy.append(lines[idx1][0], 1))
            l2 = numpy.cross(numpy.append(lines[idx2][1], 1), numpy.append(lines[idx2][0], 1))

            current_hypothesis = numpy.cross(l1, l2)
            if current_hypothesis[-1] == 0:
                continue
            idx_inliers, vote_count = self.calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh)
            if vote_count > best_vote_count:
                best_vote_count = vote_count
                best_hypothesis = current_hypothesis
                idx_best_inliers = idx_inliers
        return best_hypothesis/best_hypothesis[-1], idx_best_inliers
# ----------------------------------------------------------------------------------------------------------------------
    def get_vp(self,lines,filename_out=None,image_debug=None):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        vp1, idx_vl1 = self.run_line_ransac(lines)
        # ignore_pts = vl1
        # vp2, vl2 = self.run_line_ransac(lines, ignore_pts=ignore_pts)
        # ignore_pts = numpy.logical_or(vl1, vl2)
        # vp3, vl3 = self.run_line_ransac(lines, ignore_pts=ignore_pts)

        vp1_lines = self.reshape_lines_as_flat(lines[idx_vl1])

        if filename_out is not None:
            if image_debug is None:
                image_debug = numpy.full((self.H,self.W,3),32,dtype=numpy.uint8)
            factor = 3
            H, W = self.H, self.W
            image_ext = tools_draw_numpy.extend_view_from_image(tools_image.desaturate(image_debug), factor)
            image_ext = tools_draw_numpy.draw_lines(image_ext, tools_draw_numpy.extend_view(vp1_lines, H, W, factor), w=1)
            image_ext = tools_draw_numpy.draw_points(image_ext,tools_draw_numpy.extend_view(vp1[:2], H, W, factor),color=(255, 64, 0), w=8)
            cv2.imwrite(self.folder_out + filename_out, image_ext)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return vp1,vp1_lines

# ----------------------------------------------------------------------------------------------------------------------
    def build_BEV_by_fov_van_point(self,image, point_van_xy,cam_fov_x_deg,cam_fov_y_deg,do_rotation=True):

        if isinstance(image,str):
            image = tools_image.desaturate(cv2.imread(image), level=0)

        h_ipersp,target_BEV_W, target_BEV_H,rot_deg = tools_render_CV.get_inverce_perspective_mat_v3(image, point_van_xy,cam_fov_x_deg,do_debug=False)
        edges = numpy.array(([0, 0], [0, self.H], [self.W, 0], [self.W, self.H])).astype(numpy.float32)
        mat_R = tools_image.get_image_affine_rotation_mat(image, rot_deg, reshape=True)

        if do_rotation:
            h_ipersp = numpy.matmul(numpy.concatenate([mat_R,numpy.array([0,0,1.0]).reshape((1,-1))],axis=0),h_ipersp)
            edges_BEV = cv2.perspectiveTransform(edges.reshape((-1, 1, 2)), h_ipersp)
            target_BEV_W, target_BEV_H = numpy.max(edges_BEV.reshape((-1,2)), axis=0)

        image_BEV = cv2.warpPerspective(image, h_ipersp, (int(target_BEV_W), int(target_BEV_H)), borderValue=(32, 32, 32))
        edges_BEV = cv2.perspectiveTransform(edges.reshape((-1, 1, 2)), h_ipersp)
        center_BEV = cv2.perspectiveTransform(numpy.array((self.W / 2, self.H / 2)).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))

        lines_edges = edges_BEV.reshape((-1, 4))
        p_camera_BEV_xy = tools_render_CV.line_intersection(numpy.array(lines_edges[0]),numpy.array(lines_edges[1]))
        cam_abs_offset = p_camera_BEV_xy[1] - image_BEV.shape[0]
        center_offset = image_BEV.shape[0] - center_BEV[0][1]
        cam_height = self.evaluate_cam_height(cam_abs_offset, center_offset, cam_fov_y_deg)

        image_BEV = tools_draw_numpy.draw_points(image_BEV, center_BEV, color=self.color_markup_grid)

        return image_BEV, h_ipersp, cam_height, p_camera_BEV_xy, center_BEV.flatten(),lines_edges
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_cam_height(self,cam_abs_offset,center_offset,fov_y_deg):
        loss_g = numpy.inf
        cam_height = numpy.nan
        for h in numpy.arange(0.1, cam_abs_offset + center_offset, 0.05):
            beta  = numpy.arctan( cam_abs_offset                 /h)*180/numpy.pi
            alpha = numpy.arctan((cam_abs_offset + center_offset)/h)*180/numpy.pi
            loss = abs((alpha-beta) - fov_y_deg/2)
            if loss < loss_g:
                loss_g = loss
                cam_height = h

        #check
        # a_pitch1 = numpy.arctan( cam_abs_offset                    / cam_height) * 180 / numpy.pi
        # a_pitch2 = numpy.arctan((cam_abs_offset +   center_offset) / cam_height) * 180 / numpy.pi
        # fact = a_pitch2-a_pitch1
        # target = fov_y_deg/2

        return cam_height
# ----------------------------------------------------------------------------------------------------------------------
    def get_RAG_by_yaw(self,angle_deg,tol_min=1,tol_max=10):
        loss = abs(angle_deg)
        if loss>=tol_max:
            return self.colors_rag[-1]
        elif loss<=tol_min:
            return self.colors_rag[0]
        else:
            L = self.colors_rag.shape[0]
            return self.colors_rag[int((loss - tol_min) / (tol_max - tol_min) * L)]
# ----------------------------------------------------------------------------------------------------------------------
    def draw_grid_at_BEV(self, image_BEV, p_camera_BEV_xy, p_center_BEV_xy, lines_edges,fov_x_deg,fov_y_deg):
        cam_abs_offset = p_camera_BEV_xy[1]-image_BEV.shape[0]
        center_offset  = image_BEV.shape[0]-p_center_BEV_xy[1]
        cam_height = self.evaluate_cam_height(cam_abs_offset,center_offset,fov_y_deg)
        image_res = image_BEV

        for a_yaw in range(0, int(fov_x_deg / 2)):
            delta = p_camera_BEV_xy[1] * numpy.tan(a_yaw * numpy.pi / 180.0)
            lines_yaw = [[p_camera_BEV_xy[0] - delta, 0, p_camera_BEV_xy[0], p_camera_BEV_xy[1]],[p_camera_BEV_xy[0] + delta, 0, p_camera_BEV_xy[0], p_camera_BEV_xy[1]]]
            image_res = tools_draw_numpy.draw_lines(image_res, lines_yaw, color=self.color_markup_grid, w=1, transperency=self.transp_markup)

        res_horizontal_pitch = []
        res_vertical_yaw = []
        for a_pitch in range(1,90):
            radius = cam_height*numpy.tan(a_pitch*numpy.pi/180.0)
            center = image_BEV.shape[0]+cam_abs_offset
            row =  center-radius
            if row<0 or row>=image_BEV.shape[0]: continue
            p = (p_camera_BEV_xy[0]-radius,center-radius,p_camera_BEV_xy[0]+radius,center+radius)
            image_res = tools_draw_numpy.draw_ellipse(image_res, p, color=None, col_edge=self.color_markup_grid, transperency=self.transp_markup)
            image_res = tools_draw_numpy.draw_text(image_res,'%d' % (90-a_pitch)+u'\u00B0', (p_camera_BEV_xy[0],row-6), color_fg=self.color_markup_grid,font_size=self.font_size)
            p1 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_edges[0,2:],lines_edges[0,:-2], full_line=False)[0]
            p2 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_edges[1,2:],lines_edges[1,:-2],full_line=False)[0]
            res_horizontal_pitch.append([90-a_pitch,p1[0],p1[1],p_center_BEV_xy[0],row,p2[0],p2[1]])

            for a_yaw in range(0,int(fov_x_deg/2)):
                delta = p_camera_BEV_xy[1]*numpy.tan(a_yaw*numpy.pi/180.0)
                lines_yaw = numpy.array([[p_camera_BEV_xy[0]-delta,0,p_camera_BEV_xy[0],p_camera_BEV_xy[1]],[p_camera_BEV_xy[0]+delta,0,p_camera_BEV_xy[0],p_camera_BEV_xy[1]]])
                p1 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_yaw[0, 2:], lines_yaw[0, :-2],full_line=False)[0]
                p2 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_yaw[1, 2:], lines_yaw[1, :-2],full_line=False)[0]
                res_vertical_yaw.append([+a_yaw, p1[0], p1[1]])
                res_vertical_yaw.append([-a_yaw, p2[0], p2[1]])

        df_horizontal = pd.DataFrame(res_horizontal_pitch)
        df_vertical = pd.DataFrame(res_vertical_yaw)
        return image_res, df_horizontal,df_vertical
# ----------------------------------------------------------------------------------------------------------------------
    def draw_meters_at_BEV(self,image_BEV,p_camera_BEV_xy,pix_per_meter_BEV,pad=50):

        rows = []
        for dist_m in numpy.arange(0,200,10):
            row = p_camera_BEV_xy[1]-dist_m*pix_per_meter_BEV
            if row>0 and row<image_BEV.shape[0]:
                rows.append(row)
                image_BEV = tools_draw_numpy.draw_text(image_BEV,'%d m'%dist_m,(pad+12,row), color_fg=(255,255,255),font_size=self.font_size)

        clr = int(128)
        for r1,r2 in zip(rows[1:],rows[:-1]):
            clr = 128 if clr==255 else 255
            image_BEV[int(r1):int(r2),pad:pad+10,:] = clr

        clr = 128 if clr == 255 else 255
        image_BEV[int(rows[0]): , pad:pad+10, :] = 128
        image_BEV[:int(rows[-1]), pad:pad+10, :] = clr




        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_boxes_at_BEV(self,image_BEV,h_ipersp,df_points):

        if df_points.shape[0]==0:return image_BEV

        boxes = df_points.iloc[:, -4:].values
        lines = []
        for box in boxes:
            lines.append([box[0], box[1], box[0], box[3]])
            lines.append([box[0], box[1], box[2], box[1]])
            lines.append([box[2], box[3], box[2], box[1]])
            lines.append([box[2], box[3], box[0], box[3]])

        points_BEV = cv2.perspectiveTransform(numpy.array(lines).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape((-1,2))
        image_BEV = tools_draw_numpy.draw_points(image_BEV,points_BEV,w=4)

        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_footprints_at_BEV(self, image_BEV, h_ipersp, df_footprints,df_metadata=None):

        if df_footprints.shape[0] == 0: return image_BEV
        color = (0,0,0)
        for r in range(df_footprints.shape[0]):
            points = df_footprints.iloc[r, :].values
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r,0])
            points_BEV = cv2.perspectiveTransform(numpy.array(points).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))
            image_BEV = tools_draw_numpy.draw_contours(image_BEV, points_BEV.reshape((-1,2)), color=color, w=self.lines_width+1, transperency=0.60)

        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_grid_at_original(self, image, df_keypoints_hor,df_keypoints_ver,h_ipersp):
        image_res = image.copy()
        for r in range(df_keypoints_hor.shape[0]):
            x = df_keypoints_hor.iloc[r,[1,3,5]].values
            y = df_keypoints_hor.iloc[r,[2,4,6]].values
            label = str(df_keypoints_hor.iloc[r,0])+u'\u00B0'

            points_bev = numpy.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
            points = cv2.perspectiveTransform(points_bev.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1,2))

            lines = tools_draw_numpy.interpolate_points_by_curve(points)
            image_res = tools_draw_numpy.draw_lines(image_res, lines, color=self.color_markup_grid, transperency=self.transp_markup,w=self.width_markup_grid)
            image_res = tools_draw_numpy.draw_text(image_res, label, (points[0]+points[-1]) / 2, color_fg=self.color_markup_grid, font_size=self.font_size)


        for y in df_keypoints_ver.iloc[:,0].unique():
            df =  tools_DF.apply_filter(df_keypoints_ver,df_keypoints_ver.columns[0],y)
            points_bev = df.iloc[:,1:].values
            points = cv2.perspectiveTransform(points_bev.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1, 2))
            lines = tools_draw_numpy.interpolate_points_by_curve(points,trans=True)
            image_res = tools_draw_numpy.draw_lines(image_res, lines, color=self.color_markup_grid, transperency=self.transp_markup)

        return image_res
# ----------------------------------------------------------------------------------------------------------------------
    def drawes_boxes_at_original(self,image,df_points,df_metadata=None):
        for r in range(df_points.shape[0]):
            box = df_points.iloc[r,:].values
            image = tools_draw_numpy.draw_rect(image, box[0], box[1], box[2], box[3], color=(0,0,0), alpha_transp=1)
            if df_metadata is not None:
                label = ' '.join(['%s %.1f'%(d,v) for d,v in zip(df_metadata.columns.values,df_metadata.iloc[r,:].values)])
                color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])
                #image = tools_draw_numpy.draw_text(image,label,(box[0], box[1]), color_fg=(0,0,0),clr_bg=color,font_size=16)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def cut_boxes_at_original(self,image,df_points):
        boxes = df_points.iloc[:, -4:].values
        mask = numpy.full((image.shape[0],image.shape[1]),1,dtype=numpy.uint8)
        for box in boxes.astype(numpy.int):
            image[box[1]:box[3],box[0]:box[2]]=0
            mask[box[1]:box[3],box[0]:box[2]]=0

        return image,mask
# ----------------------------------------------------------------------------------------------------------------------
    def draw_cuboids_at_original(self,image, df_cuboids,df_metadata=None):
        lines_idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 1), (1, 5), (5, 4), (4, 0),(2, 3), (3, 7), (7, 6), (6, 2)]
        color = (0,0,0)
        for r in range(df_cuboids.shape[0]):
            cuboid = df_cuboids.iloc[r,:].values
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])
                label = ' '.join(['%s%.0f'%(d,v)+u'\u00B0'+' ' for d,v in zip(df_metadata.columns.values,df_metadata.iloc[r,:].values)])
                image = tools_draw_numpy.draw_cuboid(image, cuboid.reshape((-1, 2)), lines_idx=lines_idx, color=color,w=3)
                image = tools_draw_numpy.draw_text(image,label,(cuboid[0], cuboid[1]+self.font_size), color_fg=(0,0,0),clr_bg=color,font_size=self.font_size)
            else:
                image = tools_draw_numpy.draw_cuboid(image, cuboid.reshape((-1, 2)), lines_idx=lines_idx, color=color,w=self.lines_width)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_BEVs_folder(self,vp_ver,vp_hor,fov_x_deg,fov_y_deg,pix_per_meter_BEV,folder_in,list_of_masks='*.jpg',df_boxes=None,limit=5000):
        tools_IO.remove_files(self.folder_out)
        image_bg = cv2.imread(folder_in+tools_IO.get_filenames(folder_in, list_of_masks)[0])

        for filename in tools_IO.get_filenames(folder_in, list_of_masks)[:limit]:
            print(filename)
            image = cv2.imread(folder_in+filename)
            df = tools_DF.apply_filter(df_boxes,df_boxes.columns[0],filename)
            image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.build_BEV_by_fov_van_point(image_bg, vp_ver, fov_x_deg, fov_y_deg, do_rotation=True)
            image_BEV, df_keypoints_pitch,df_vertical = self.draw_grid_at_BEV(image_BEV, p_camera_BEV_xy, p_center_BEV_xy, lines_edges, fov_x_deg, fov_y_deg)
            image_BEV = self.draw_meters_at_BEV(image_BEV,p_camera_BEV_xy,pix_per_meter_BEV,pad=75)

            df_objects = self.evaluate_objects(h_ipersp, df, vp_ver, vp_hor, cam_height_px, p_camera_BEV_xy,p_center_BEV_xy)
            df_footprints = df_objects.iloc[:,2:10]
            df_cuboids =  df_objects.iloc[:,2:18]
            df_metadata = df_objects.iloc[:,-2:]

            image_BEV = self.draw_footprints_at_BEV(image_BEV, h_ipersp,df_footprints,df_metadata)
            image_res = self.draw_cuboids_at_original(image, df_cuboids,df_metadata)
            image_res = self.drawes_boxes_at_original(image_res, df.iloc[:,3:8])
            image_res = self.draw_grid_at_original(image_res, df_keypoints_pitch, df_vertical, h_ipersp)
            image_res = tools_draw_numpy.draw_text(image_res,'cam fov=%.f'%fov_x_deg+u'\u00B0'+'\nheight=%.1f m'%(cam_height_px/pix_per_meter_BEV),(0,self.H-50), color_fg=(255,255,255),clr_bg=None,font_size=40)

            cv2.imwrite(self.folder_out + filename, tools_image.hstack_images(image_res[:,:-150],image_BEV[:,75:]))

        return
# ----------------------------------------------------------------------------------------------------------------------
#     def fetch_pose_cuboids(self,df_boxes,vp_ver,vp_hor,fov_x_deg,fov_y_deg,h_ipersp, cam_height,p_camera_BEV_xy,p_center_BEV_xy):
#
#         for filename in df_boxes.iloc[:,0].unique():
#             df = tools_DF.apply_filter(df_boxes,df_boxes.columns[0],filename)
#             df_cuboids = self.evaluate_objects(h_ipersp, df, vp_ver, vp_hor, cam_height, p_camera_BEV_xy)
#             df_foorprints = df_cuboids.iloc[:, :9]
#         return
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_objects(self, h_ipersp, df_points, vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,image=None):

        if df_points.shape[0]==0:return pd.DataFrame([])

        res = []
        for box in df_points.iloc[:, -4:].values:
            point_anchor = numpy.array((min(box[0],box[2]),max(box[1],box[3])))
            line_anchor  = numpy.array((max(box[0],box[2]),box[1],max(box[0],box[2]),box[3]))
            vec_width    = numpy.array((vp_hor[0]-point_anchor[0], vp_hor[1]-point_anchor[1]))
            vec_width = -vec_width/numpy.linalg.norm(vec_width)
            width_max  = (max(box[0], box[2])-min(box[0],box[2]))
            height_max = (max(box[1], box[3])-min(box[1],box[3]))

            ratio_best = None
            for fraction in numpy.linspace(0.01,1,100,endpoint=True):
                width = width_max*fraction
                point_edge1 = point_anchor + vec_width*width
                line_van_ver1 = numpy.array((vp_ver[0], vp_ver[1],point_edge1[0],point_edge1[1]))
                point_edge2 = tools_render_CV.line_intersection(line_van_ver1,line_anchor)
                if point_edge2[1]<box[1] or point_edge2[1]>box[3]:
                    point_edge2[1] = box[1]+(height_max*fraction)
                    point_edge1 = (box[2], box[3])

                line_van_ver_anchor = numpy.array((vp_ver[0], vp_ver[1], point_anchor[0], point_anchor[1]))
                line_van_hor_edge2 = numpy.array((vp_hor[0], vp_hor[1], point_edge2[0], point_edge2[1]))
                point_edge3 = tools_render_CV.line_intersection(line_van_ver_anchor, line_van_hor_edge2)
                points_BEV = cv2.perspectiveTransform(numpy.array([point_anchor,point_edge1,point_edge2,point_edge3]).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))
                cuboid_h = min(point_edge2[1],point_edge3[1])-min(box[1],box[3])
                ratio = abs(points_BEV[0][1]-points_BEV[-1][1])/(abs(points_BEV[0][0]-points_BEV[1][0])+1e-4)
                #print(ratio)
                # im = tools_draw_numpy.draw_lines(image, [line_anchor, line_van_ver1])
                # im = tools_draw_numpy.draw_points(im, [point_anchor, point_edge1,point_edge2,point_edge3])
                # cv2.imwrite(self.folder_out + 'xxx.png', im)

                if ratio_best is None or abs(ratio-self.ratio_target)<abs(ratio_best-self.ratio_target):
                    ratio_best = ratio
                    footprint = numpy.array([point_anchor,point_edge1,point_edge2,point_edge3]).reshape(1,-1)
                    roofprint = footprint - numpy.array([(0,cuboid_h),(0,cuboid_h),(0,cuboid_h),(0,cuboid_h)]).reshape(1,-1)
                    yaw1 = self.get_angle_deg((points_BEV[0][0],points_BEV[0][1],points_BEV[3][0],points_BEV[3][1]))
                    yaw2 = self.get_angle_deg((points_BEV[1][0],points_BEV[1][1],points_BEV[2][0],points_BEV[2][1]))
                    yaw_ego = (yaw1+yaw2)/2
                    yaw_cam = numpy.arctan((0.5*(points_BEV[0][0]+points_BEV[1][0])-p_center_BEV_xy[0])/(p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1])))*180/numpy.pi
                    yaw_res = -yaw_ego+yaw_cam
                    pitch_cam  = 90-numpy.arctan((p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1]))/cam_height)*180/numpy.pi
                    angles = numpy.array([abs(yaw_res), pitch_cam]).reshape((1, -1))

                # image_cand = tools_draw_numpy.draw_points(image, [point_anchor, point_edge1, point_edge2, point_edge3])
                # image_cand = tools_draw_numpy.draw_contours(image_cand,numpy.array([point_anchor,point_edge1,point_edge2,point_edge3]), color=(0,0,200),transperency=0.75)
                # cv2.imwrite(self.folder_out+'F_%02d_%02d.png'%(0,width),image_cand)

            res.append(numpy.concatenate((footprint,roofprint,angles),axis=1).flatten())

        df_cuboids = pd.DataFrame(res,columns=['cuboid%02d'%i for i in range(16)]+['H','V'])
        df_points.reset_index(drop=True, inplace=True)
        df_cuboids.reset_index(drop=True, inplace=True)
        df_objects = pd.concat([df_points.iloc[:,:2],df_cuboids],axis=1)

        return df_objects
# ----------------------------------------------------------------------------------------------------------------------
    def remove_bg(self,df_boxes,folder_in,list_of_masks='*.jpg',limit=50):
        image_S = numpy.zeros((self.H,self.W,3),dtype=numpy.long)
        image_C = numpy.zeros((self.H,self.W  ),dtype=numpy.long)

        for filename in tools_IO.get_filenames(folder_in, list_of_masks)[:limit]:
            image = cv2.imread(folder_in+filename)
            df = tools_DF.apply_filter(df_boxes,df_boxes.columns[0],filename)
            image_cut, mask = self.cut_boxes_at_original(image, df)
            image_S+=image_cut
            image_C+=mask

        image_S[:, :, [0,1,2]:] = image_S[:, :, [0,1,2]:] / image_C
        return image_S
# ----------------------------------------------------------------------------------------------------------------------


