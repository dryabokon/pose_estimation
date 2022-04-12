
import json
import os
import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(1, './tools/')
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_DF
import tools_video
# ----------------------------------------------------------------------------------------------------------------------
import utils_vanishing_point
# ----------------------------------------------------------------------------------------------------------------------
class PoseEstimation_pipelines:
    def __init__(self,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)
        self.folder_out = folder_out
        self.U_VP = utils_vanishing_point.detector_VP(folder_out)
        self.dct_res = {}
        return
# ----------------------------------------------------------------------------------------------------------------------
    def save_dict(self,dct,filename_out):
        with open(self.folder_out+filename_out, 'w') as f:
            json.dump(dct, f, indent=' ')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def stream_to_images(self,URL, folder_out=None, limit=100):
        folder_out = self.folder_out if folder_out is None else folder_out
        tools_video.extract_frames_v2(URL, folder_out, prefix='', start_frame=0, end_frame=limit, step=1, scale=1,silent=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def evaluate_VP_FOV_height(self, URL, do_debug=False):

        self.dct_res['stream'] = URL

        tools_IO.remove_files(self.folder_out, '*.png,*.jpg')
        self.stream_to_images(URL,limit=100)

        folder_in = self.folder_out
        filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        if len(filenames)==0:
            return

        image = cv2.imread(self.folder_out + filenames[0])

        #lines_ver = U_VP.get_lines_ver_candidates_dynamic(folder_in,do_debug=True)
        lines_ver = self.U_VP.get_lines_ver_candidates_static(folder_in,do_debug=do_debug)
        vp_ver, lines_vp_ver = self.U_VP.get_vp(self.U_VP.reshape_lines_as_paired(lines_ver), filename_out='VP_ver.png', image_debug=image)
        self.dct_res['vp_ver_x'],self.dct_res['vp_ver_y'] = vp_ver[0],vp_ver[1]

        lines_hor = self.U_VP.get_lines_hor_candidates_static(folder_in,do_debug=do_debug)
        vp_hor, lines_vp_hor = self.U_VP.get_vp(self.U_VP.reshape_lines_as_paired(lines_hor),filename_out='VP_hor.png', image_debug=image)
        self.dct_res['vp_hor_x'], self.dct_res['vp_hor_y'] = vp_hor[0], vp_hor[1]

        focal_lenth = self.U_VP.get_focal_length([vp_ver, vp_hor])
        fov_x_deg = 2 * numpy.arctan(0.5 * self.U_VP.H / focal_lenth) * 180 / numpy.pi
        fov_y_deg = fov_x_deg * image.shape[0] / image.shape[1]
        self.dct_res['fov_x_deg'] = fov_x_deg
        self.dct_res['fov_y_deg'] = fov_y_deg

        image_BEV, h_ipersp, cam_height, p_camera_BEV_xy, center_BEV,lines_edges = self.U_VP.build_BEV_by_fov_van_point(image, vp_ver, fov_x_deg,fov_y_deg, do_rotation=True)
        self.dct_res['cam_height_pixels'] = cam_height

        tools_IO.remove_files(self.folder_out, '*.jpg')
        if do_debug:
            cv2.imwrite(self.folder_out+'BEV.png',image_BEV)

        self.save_dict(self.dct_res, filename_out='result.json')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipeline_obj_detection_folder(self):
        from detector import detector_TF_Zoo
        D = detector_TF_Zoo.detector_TF_Zoo(folder_out)
        D.process_folder(folder_in)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipeline_post_process_detection(self,size_min=20,size_max=900,th_min_iou=0.25):
        df_boxes = pd.read_csv(folder_in + 'df_boxes.csv')

        IDs = df_boxes.iloc[:, 0].unique()
        res,res_bad = [],[]
        for i in range(IDs.shape[0]):
            df_cur = tools_DF.apply_filter(df_boxes, df_boxes.columns[0], IDs[i])
            df_prev = tools_DF.apply_filter(df_boxes, df_boxes.columns[0],IDs[i-1]) if i>0 else df_cur
            df_next = tools_DF.apply_filter(df_boxes, df_boxes.columns[0],IDs[i+1]) if i<IDs.shape[0]-1 else df_cur
            for b,box_cur in enumerate(df_cur.iloc[:,3:].values):
                if abs(box_cur[0]-box_cur[2])<size_min or abs(box_cur[0]-box_cur[2])>size_max:
                    res_bad.append(df_cur.iloc[b,:].values)
                    continue
                if abs(box_cur[1]-box_cur[3])<size_min or abs(box_cur[1]-box_cur[3])>size_max:
                    res_bad.append(df_cur.iloc[b, :].values)
                    continue
                iou = 0
                for box in df_prev.iloc[:,3:].values:iou = max(iou,U_VP.iou(box_cur,box))
                for box in df_next.iloc[:,3:].values:iou = max(iou,U_VP.iou(box_cur,box))
                if iou>th_min_iou:
                    res.append(df_cur.iloc[b,:].values)
                else:
                    res_bad.append(df_cur.iloc[b, :].values)

        df_res = pd.DataFrame(res,columns=df_boxes.columns)
        df_res.to_csv(folder_out+'df_boxes_filtered.csv',index=False)

        df_res_bad = pd.DataFrame(res_bad,columns=df_boxes.columns)
        df_res_bad.to_csv(folder_out+'df_boxes_bad.csv',index=False)

        print(df_boxes.shape[0],df_res_bad.shape[0])

        return df_res
# ----------------------------------------------------------------------------------------------------------------------
    def pipeline_bg_rem(self):
        df_boxes = pd.read_csv(folder_in+'df_boxes.csv')
        image_bg = self.U_VP.remove_bg(df_boxes,folder_in)
        cv2.imwrite(folder_out+'bg.jpg',image_bg)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipeline_draw(self):
        df_boxes = pd.read_csv(folder_in+'df_boxes_filtered.csv')
        self.U_VP.draw_BEVs_folder(vp_ver,vp_hor,fov_x_deg,fov_y_deg,pix_per_meter_BEV,folder_in,df_boxes=df_boxes)
        return
# ----------------------------------------------------------------------------------------------------------------------

