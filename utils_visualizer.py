import math
import pandas as pd
import numpy
import cv2
# ---------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_IO
import tools_image
import tools_plot_v2
import tools_render_CV
import tools_render_GL
import tools_time_profiler
import tools_DF
from CV import tools_pr_geom
from CV import tools_Skeletone
from CV import tools_calibrator
from CV import tools_vanishing
# ---------------------------------------------------------------------------------------------------------------------
class visualizer:
    def __init__(self, folder_out,W=None,H=None):
        self.folder_out = folder_out
        self.Ske = tools_Skeletone.Skelenonizer(folder_out=folder_out)
        self.P = tools_plot_v2.Plotter(folder_out=folder_out)

        self.VP = tools_vanishing.detector_VP(folder_out)
        self.TP = tools_time_profiler.Time_Profiler()
        self.Calibrator = tools_calibrator.Calibrator()

        self.H = H
        self.W = W

        self.taret_ratio_L_W = 4685/ 1814   #vehicle ratio
        self.taret_ratio_H_W = 1449 / 1814  #vehicle ratio
        self.mean_vehicle_length = 4.685  #meters

        self.tol_deg_hor_line = 10
        self.color_markup_grid = (0,0,0)
        self.width_markup_grid = 2
        self.lines_width = 2
        self.color_markup_cuboid = tools_draw_numpy.color_black
        self.transp_markup = 0.85
        self.colors_rag = tools_draw_numpy.get_colors(255,colormap = 'nipy_spectral')[120:240]
        self.font_size = 28
        self.filename_vehicle_3d_obj = './SUV1.obj'

        return
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
    def get_angle_deg(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            angle = 0
        else:
            angle = 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
        return angle
# ----------------------------------------------------------------------------------------------------------------------
    def draw_grid_at_BEV(self, image_BEV, p_camera_BEV_xy, p_center_BEV_xy, lines_edges,fov_x_deg,fov_y_deg):
        cam_abs_offset = p_camera_BEV_xy[1]-image_BEV.shape[0]
        center_offset  = image_BEV.shape[0]-p_center_BEV_xy[1]
        cam_height = self.VP.evaluate_cam_height(cam_abs_offset,center_offset,fov_y_deg)
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


            p1 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_edges[0,2:],lines_edges[0,:-2], full_line=False)
            p2 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_edges[1,2:],lines_edges[1,:-2],full_line=False)
            if len(p1)>0 and len(p2)>0:
                res_horizontal_pitch.append([90-a_pitch,p1[0][0],p1[0][1],p_center_BEV_xy[0],row,p2[0][0],p2[0][1]])

            for a_yaw in range(0,int(fov_x_deg/2)):
                delta = p_camera_BEV_xy[1]*numpy.tan(a_yaw*numpy.pi/180.0)
                lines_yaw = numpy.array([[p_camera_BEV_xy[0]-delta,0,p_camera_BEV_xy[0],p_camera_BEV_xy[1]],[p_camera_BEV_xy[0]+delta,0,p_camera_BEV_xy[0],p_camera_BEV_xy[1]]])
                p1 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_yaw[0, 2:], lines_yaw[0, :-2],full_line=False)
                p2 = tools_render_CV.circle_line_intersection(p_camera_BEV_xy, radius, lines_yaw[1, 2:], lines_yaw[1, :-2],full_line=False)
                if len(p1) > 0 and len(p2) > 0:
                    res_vertical_yaw.append([+a_yaw, p1[0][0], p1[0][1]])
                    res_vertical_yaw.append([-a_yaw, p2[0][0], p2[0][1]])

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
        if len(rows)>0:
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
    def draw_footprints_at_BEV(self, image_BEV, h_ipersp, df_footprints,color = (0,0,0),df_metadata=None):

        if df_footprints.shape[0] == 0: return image_BEV

        for r in range(df_footprints.shape[0]):
            points = df_footprints.iloc[r, :].values
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r,0])
            points_BEV = cv2.perspectiveTransform(numpy.array(points).astype(numpy.float32).reshape((-1, 1, 2)),h_ipersp).reshape((-1, 2))
            image_BEV = tools_draw_numpy.draw_contours(image_BEV, points_BEV.reshape((-1,2)), color=color, w=self.lines_width+1, transperency=0.60)

        return image_BEV
# ----------------------------------------------------------------------------------------------------------------------
    def draw_points_at_BEV(self, image_BEV, df_points, color=(0, 0, 0),w=16,df_metadata=None):

        if df_points.shape[0] == 0: return image_BEV

        for r in range(df_points.shape[0]):
            points_BEV = df_points.iloc[r, :].values
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])
            image_BEV = tools_draw_numpy.draw_points(image_BEV, points_BEV.reshape((-1, 2)), color=color, w=w)

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
    def drawes_boxes_at_original(self,image,df_points,color=(0,0,0),w=1,df_metadata=None):
        for r in range(df_points.shape[0]):
            box = df_points.iloc[r,:].values
            label = None
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])
                label = ' '.join(['%s%.0f' % (d, v) + u'\u00B0' + ' ' for d, v in zip(df_metadata.columns.values[:2], df_metadata.iloc[r, :2].values)])
            image = tools_draw_numpy.draw_rect(image, box[0], box[1], box[2], box[3], color=color, w=w,alpha_transp=1,font_size=self.font_size,label=label)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def cut_boxes_at_original(self,image,df_points,pad=10):

        H,W = image.shape[:2]
        boxes = df_points.iloc[:, -4:].values
        mask = numpy.full((image.shape[0],image.shape[1]),1,dtype=numpy.uint8)
        for box in boxes.astype(numpy.int):
            image[max(0,box[1]-pad):min(box[3]+pad,W),max(0,box[0]-pad):min(box[2]+pad,H)]=0
            mask [max(0,box[1]-pad):min(box[3]+pad,W),max(0,box[0]-pad):min(box[2]+pad,H)]=0

        return image,mask
# ----------------------------------------------------------------------------------------------------------------------
    def draw_cuboids_at_original(self,image, df_cuboids,idx_mode=0,color=(0,0,0),df_metadata=None):

        for r in range(df_cuboids.shape[0]):
            cuboid = df_cuboids.iloc[r,:].values
            if df_metadata is not None:
                color = self.get_RAG_by_yaw(df_metadata.iloc[r, 0])

                label_a = ' '.join(['%s%.0f'%(d,v)+u'\u00B0'+' ' for d,v in zip(df_metadata.columns.values[:2], df_metadata.iloc[r,:2].values)])
                label_b = ' '.join(['%s%.1f'%(d,v)          +' ' for d,v in zip(df_metadata.columns.values[2:], df_metadata.iloc[r,2:].values)])
                label = label_a+' '+label_b

                image = tools_draw_numpy.draw_cuboid(image, cuboid.reshape((-1, 2)), idx_mode=idx_mode,color=color,w=3)
                xpos = min(cuboid[[0,2,4,6]])
                ypos = max(cuboid[[1,3,5,7]])
                image = tools_draw_numpy.draw_text(image,label,(xpos, ypos+self.font_size), color_fg=(0,0,0),clr_bg=color,font_size=self.font_size)
            else:
                image = tools_draw_numpy.draw_cuboid(image, cuboid.reshape((-1, 2)),idx_mode=idx_mode,color=color,w=self.lines_width)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_contour_at_original(self,image, points_2d,color=None):
        for pp in points_2d:
            image = tools_draw_numpy.draw_convex_hull(image, pp, color=color,transperency=0.50)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def box_to_footprint_look_upleft(self,box,vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,h_ipersp):
        point_bottom_left = numpy.array((min(box[0], box[2]), max(box[1], box[3])))
        line_van_ver_left = numpy.array((min(box[0], box[2]), max(box[1], box[3]), vp_ver[0], vp_ver[1]))
        footprint, roofprint, angles,points_BEV_best = None,None,None,None
        ratio_best = None
        for point_top_right_y in range(min(box[1], box[3]), max(box[1], box[3])):
            point_top_right = (max(box[0], box[2]), point_top_right_y)
            line_van_hor_top = (vp_hor[0], vp_hor[1], point_top_right[0], point_top_right[1])
            line_van_hor_bottom = (vp_hor[0], vp_hor[1], point_bottom_left[0], point_bottom_left[1])
            point_top_left = tools_render_CV.line_intersection(line_van_hor_top, line_van_ver_left)
            line_van_ver_right = numpy.array((point_top_right[0], point_top_right[1], vp_ver[0], vp_ver[1]))
            point_bottom_right = tools_render_CV.line_intersection(line_van_hor_bottom, line_van_ver_right)

            points_BEV = cv2.perspectiveTransform(numpy.array([point_bottom_left, point_bottom_right, point_top_right, point_top_left]).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))
            cuboid_h = min(point_top_right[1], point_top_left[1]) - min(box[1], box[3])
            ratio = abs(points_BEV[0][1] - points_BEV[-1][1]) / (abs(points_BEV[0][0] - points_BEV[1][0]) + 1e-4)
            if ratio_best is None or abs(ratio - self.taret_ratio_L_W) < abs(ratio_best - self.taret_ratio_L_W):
                ratio_best = ratio
                footprint = numpy.array([point_bottom_left, point_bottom_right, point_top_right, point_top_left]).reshape(1, -1)
                roofprint = footprint - numpy.array([(0, cuboid_h), (0, cuboid_h), (0, cuboid_h), (0, cuboid_h)]).reshape(1, -1)
                yaw1 = self.get_angle_deg((points_BEV[0][0], points_BEV[0][1], points_BEV[3][0], points_BEV[3][1]))
                yaw2 = self.get_angle_deg((points_BEV[1][0], points_BEV[1][1], points_BEV[2][0], points_BEV[2][1]))
                yaw_ego = (yaw1 + yaw2) / 2
                yaw_cam = numpy.arctan((0.5 * (points_BEV[0][0] + points_BEV[1][0]) - p_center_BEV_xy[0]) / (p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1]))) * 180 / numpy.pi
                # yaw_res = -yaw_ego+yaw_cam # with ego-compensation
                yaw_res = yaw_ego
                pitch_cam = 90 - numpy.arctan((p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1])) / cam_height) * 180 / numpy.pi
                angles = numpy.array([abs(yaw_res), pitch_cam]).reshape((1, -1))
                points_BEV_best = points_BEV.copy().reshape((1, -1))

                # image_cand = tools_draw_numpy.draw_points(image, [point_bottom_left,point_bottom_right,point_top_right,point_top_left])
                # image_cand = tools_draw_numpy.draw_contours(image_cand,numpy.array([point_bottom_left,point_bottom_right,point_top_right,point_top_left]), color=(0,0,200),transperency=0.75)
                # cv2.imwrite(self.folder_out+'F_%02d_%03d.png'%(0,point_top_right[1]),image_cand)

        cols = ['cuboid%02d' % i for i in range(16)] + ['yaw_ego_deg', 'pitch_cam'] + ['p_bev%02d' % i for i in range(8)]
        df = pd.DataFrame(numpy.concatenate((footprint, roofprint, angles, points_BEV_best), axis=1).reshape((1, -1)),columns=cols)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def box_to_footprint_look_upright(self,box,vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,h_ipersp):

        point_bottom_right2 = numpy.array((max(box[0], box[2]), max(box[1], box[3])))
        line_van_ver_right2 = numpy.array((max(box[0], box[2]), max(box[1], box[3]), vp_ver[0], vp_ver[1]))
        footprint, roofprint, angles,points_BEV_best,dims = None,None,None,None,None
        ratio_best = None
        for point_top_left_y2 in range(min(box[1], box[3]), max(box[1], box[3])):
            point_top_left2 = (min(box[0], box[2]), point_top_left_y2)
            line_van_hor_top = (vp_hor[0], vp_hor[1], point_top_left2[0], point_top_left2[1])
            line_van_hor_bottom = (vp_hor[0], vp_hor[1], point_bottom_right2[0], point_bottom_right2[1])
            point_top_right2 = tools_render_CV.line_intersection(line_van_hor_top, line_van_ver_right2)
            line_van_ver_left2 = numpy.array((point_top_left2[0], point_top_left2[1], vp_ver[0], vp_ver[1]))
            point_bottom_left2 = tools_render_CV.line_intersection(line_van_hor_bottom, line_van_ver_left2)

            points_BEV = cv2.perspectiveTransform(numpy.array([point_bottom_right2, point_bottom_left2, point_top_left2, point_top_right2]).astype(numpy.float32).reshape((-1, 1, 2)), h_ipersp).reshape((-1, 2))
            cuboid_h = min(point_top_left2[1], point_top_right2[1]) - min(box[1], box[3])
            ratio = abs(points_BEV[0][1] - points_BEV[-1][1]) / (abs(points_BEV[0][0] - points_BEV[1][0]) + 1e-4)
            if ratio_best is None or abs(ratio - self.taret_ratio_L_W) < abs(ratio_best - self.taret_ratio_L_W):
                ratio_best = ratio
                footprint = numpy.array([point_bottom_right2, point_bottom_left2, point_top_left2, point_top_right2]).reshape(1, -1)
                roofprint = footprint - numpy.array([(0, cuboid_h), (0, cuboid_h), (0, cuboid_h), (0, cuboid_h)]).reshape(1, -1)
                yaw1 = self.get_angle_deg((points_BEV[0][0], points_BEV[0][1], points_BEV[3][0], points_BEV[3][1]))
                yaw2 = self.get_angle_deg((points_BEV[1][0], points_BEV[1][1], points_BEV[2][0], points_BEV[2][1]))
                yaw_ego = (yaw1 + yaw2) / 2
                #yaw_cam = numpy.arctan((0.5 * (points_BEV[0][0] + points_BEV[1][0]) - p_center_BEV_xy[0]) / (p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1]))) * 180 / numpy.pi
                yaw_res = yaw_ego
                pitch_cam = 90 - numpy.arctan((p_camera_BEV_xy[1] - 0.5 * (points_BEV[0][1] + points_BEV[1][1])) / cam_height) * 180 / numpy.pi
                angles = numpy.array([abs(yaw_res), pitch_cam]).reshape((1, -1))
                points_BEV_best = points_BEV.copy().reshape((1, -1))

        cols = ['cuboid%02d' % i for i in range(16)] + ['yaw_ego_deg', 'pitch_cam'] + ['p_bev%02d' % i for i in range(8)]
        df = pd.DataFrame(numpy.concatenate((footprint, roofprint, angles, points_BEV_best), axis=1).reshape((1, -1)),columns=cols)
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_objects(self, h_ipersp, df_boxes, vp_ver, vp_hor, cam_height, p_camera_BEV_xy, p_center_BEV_xy):

        if df_boxes.shape[0]==0:return pd.DataFrame([])
        df_cuboids_all = pd.DataFrame([])
        for box in df_boxes.iloc[:, -4:].values:
            if vp_ver[0]>min(box[0], box[2]):
                df_cuboids = self.box_to_footprint_look_upleft(box, vp_ver, vp_hor, cam_height, p_camera_BEV_xy,p_center_BEV_xy,h_ipersp)
            else:
                df_cuboids = self.box_to_footprint_look_upright(box, vp_ver, vp_hor, cam_height,p_camera_BEV_xy, p_center_BEV_xy,h_ipersp)
            df_cuboids_all = df_cuboids_all.append(df_cuboids,ignore_index=True)

        df_boxes.reset_index(drop=True, inplace=True)
        df_cuboids_all.reset_index(drop=True, inplace=True)
        df_objects = pd.concat([df_boxes, df_cuboids_all], axis=1)

        return df_objects
# ----------------------------------------------------------------------------------------------------------------------
    def remove_bg(self,df_boxes,folder_in,list_of_masks='*.jpg',limit=50):

        image_S = numpy.zeros((self.H,self.W,3),dtype=numpy.long)
        image_C = numpy.zeros((self.H,self.W  ),dtype=numpy.long)

        for filename in tools_IO.get_filenames(folder_in, list_of_masks)[:limit]:
            image = cv2.imread(folder_in+filename)
            if df_boxes.shape[0]>0:
                df = tools_DF.apply_filter(df_boxes,df_boxes.columns[0],filename)
            else:
                df=df_boxes
            image_cut, mask = self.cut_boxes_at_original(image, df)
            image_S+=image_cut
            image_C+=mask

        image_S[:, :, 0] = image_S[:, :, 0] / image_C
        image_S[:, :, 1] = image_S[:, :, 1] / image_C
        image_S[:, :, 2] = image_S[:, :, 2] / image_C
        return image_S.astype(numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
    def construct_cuboid(self, dim, rvec, tvec):
        dw, dh, dl = dim[0]/2, dim[1]/2, dim[2]/2
        X = [[-dw, -dh, -dl],[+dw, -dh, -dl],[+dw, -dh, +dl],[-dw, -dh, +dl],[-dw, +dh, -dl],[+dw, +dh, -dl],[+dw, +dh, +dl],[-dw, +dh, +dl]]
        X = numpy.array(X)
        X = tools_pr_geom.apply_rotation(rvec, X)
        X = tools_pr_geom.apply_translation(tvec, X)
        return X
# ----------------------------------------------------------------------------------------------------------------------
    def get_vehicle_dims(self,df_objects,r,p_center_BEV_xy,pix_per_meter_BEV):
        dim1 = numpy.linalg.norm(df_objects[['p_bev00', 'p_bev01']].iloc[r].values - df_objects[['p_bev02', 'p_bev03']].iloc[r].values) / pix_per_meter_BEV
        dim2 = numpy.linalg.norm(df_objects[['p_bev02', 'p_bev03']].iloc[r].values - df_objects[['p_bev04', 'p_bev05']].iloc[r].values) / pix_per_meter_BEV
        dimW, dimL = min(dim1, dim2), max(dim1, dim2)
        dimH = self.mean_vehicle_length * self.taret_ratio_H_W / self.taret_ratio_L_W
        dims = (dimW, dimH, dimL)

        rvec_car = (0, 0, numpy.pi * df_objects['yaw_ego_deg'].iloc[r] / 180)
        centroid_x = df_objects[['p_bev00', 'p_bev02', 'p_bev04', 'p_bev06']].iloc[r].mean()
        centroid_y = df_objects[['p_bev01', 'p_bev03', 'p_bev05', 'p_bev07']].iloc[r].mean()
        centroid_x_m = -(p_center_BEV_xy[0] - centroid_x) / pix_per_meter_BEV
        centroid_y_m = (p_center_BEV_xy[1] - centroid_y) / pix_per_meter_BEV
        tvec_car = numpy.array((centroid_x_m, +dimH / 2, centroid_y_m))
        return dims, rvec_car, tvec_car
# ----------------------------------------------------------------------------------------------------------------------
    def get_cuboids(self,df_objects,p_center_BEV_xy,pix_per_meter_BEV,rvec, tvec, camera_matrix_3x3, mat_trns):

        cuboids_GL  = []
        for r in range(df_objects.shape[0]):
            dims, rvec_car, tvec_car = self.get_vehicle_dims(df_objects,r,p_center_BEV_xy,pix_per_meter_BEV)
            points_3d = self.construct_cuboid(dims, rvec_car, tvec_car)
            points_2d = tools_render_GL.project_points_rvec_tvec_GL(points_3d, rvec, tvec, camera_matrix_3x3, mat_trns)
            cuboids_GL.append(points_2d)

        return pd.DataFrame(numpy.array(cuboids_GL).reshape((-1,16)))

# ----------------------------------------------------------------------------------------------------------------------
    def draw_BEVs_cars_folder(self, df_objects_all, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, folder_in, list_of_masks='*.jpg', image_clear_bg=None):

        image_bg = image_clear_bg if image_clear_bg is not None else cv2.imread(folder_in+tools_IO.get_filenames(folder_in, list_of_masks)[0])
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.VP.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg,vp_ver,vp_hor, do_rotation=True)
        image_BEV, df_keypoints_pitch, df_vertical = self.draw_grid_at_BEV(image_BEV, p_camera_BEV_xy, p_center_BEV_xy,lines_edges, fov_x_deg, fov_y_deg)
        image_BEV = self.draw_meters_at_BEV(image_BEV, p_camera_BEV_xy, pix_per_meter_BEV, pad=0)
        df_objects_all[['L','W','H']]/=pix_per_meter_BEV

        prefix = '' if list_of_masks=='*.jpg' else '%04d_'%int(fov_x_deg*100)
        for filename in tools_IO.get_filenames(folder_in, list_of_masks):
            image = cv2.imread(folder_in+filename)

            df_objects = tools_DF.apply_filter(df_objects_all, 'ID', filename)

            df_boxes      = df_objects.iloc[:,3:7]
            df_footprints = df_objects.iloc[:,7:7+ 8]
            cuboids_orig    = df_objects.iloc[:,7:7+16]
            df_metadata = df_objects[['yaw_cam_car', 'pitch_cam','L','W','H']]
            image_BEV_local = self.draw_footprints_at_BEV(image_BEV, h_ipersp, df_footprints, df_metadata=df_metadata)

            image_res = self.draw_grid_at_original(image, df_keypoints_pitch, df_vertical, h_ipersp)
            image_res = self.draw_cuboids_at_original(image_res, cuboids_orig,df_metadata=df_metadata)
            image_res = self.drawes_boxes_at_original(image_res, df_boxes)

            image_res = tools_draw_numpy.draw_text(image_res,'cam fov=%.1f'%fov_x_deg+u'\u00B0'+'\nheight=%.2f m'%(cam_height_px/pix_per_meter_BEV),(0,self.H-50), color_fg=(255,255,255),clr_bg=None,font_size=40)
            cv2.imwrite(self.folder_out + prefix + filename, tools_image.hstack_images(image_res,image_BEV_local))
            ii=0

        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_BEVs_lps_folder(self,df_objects_all, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, folder_in, list_of_masks='*.jpg'):

        image_bg = cv2.imread(folder_in + tools_IO.get_filenames(folder_in, list_of_masks)[0])
        image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.VP.build_BEV_by_fov_van_point(image_bg, fov_x_deg, fov_y_deg, vp_ver, vp_hor, do_rotation=True)
        image_BEV, df_keypoints_pitch, df_vertical = self.draw_grid_at_BEV(image_BEV, p_camera_BEV_xy, p_center_BEV_xy,lines_edges, fov_x_deg, fov_y_deg)
        image_BEV = tools_image.desaturate(self.draw_meters_at_BEV(image_BEV, p_camera_BEV_xy, pix_per_meter_BEV, pad=0))
        image_BEV = self.draw_meters_at_BEV(image_BEV, p_camera_BEV_xy, pix_per_meter_BEV, pad=0)

        prefix = '' if list_of_masks == '*.jpg' else '%04d_' % int(fov_x_deg * 100)

        for filename in tools_IO.get_filenames(folder_in, list_of_masks):
            image = tools_image.desaturate(cv2.imread(folder_in+filename))
            df_objects = tools_DF.apply_filter(df_objects_all, 'ID', filename)
            df_metadata = df_objects[['yaw_cam_car', 'pitch_cam']]

            image_res = self.draw_grid_at_original(image, df_keypoints_pitch, df_vertical, h_ipersp)
            image_res = self.drawes_boxes_at_original(image_res, df_objects.iloc[:, 3:7],color=(0,0,200),w=3,df_metadata=df_metadata)
            image_BEV_local = self.draw_points_at_BEV(image_BEV, df_objects.iloc[:, 9:11], df_metadata=df_metadata)
            cv2.imwrite(self.folder_out + prefix + filename, tools_image.hstack_images(image_res,image_BEV_local))
            ii=0

        return
# ----------------------------------------------------------------------------------------------------------------------