import os
import cv2
import numpy
from shutil import copyfile
import inspect
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(1, './tools/')
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_DF
import tools_video
import tools_time_profiler
from CV import tools_vanishing
# ----------------------------------------------------------------------------------------------------------------------
import utils_visualizer
# ----------------------------------------------------------------------------------------------------------------------
class PoseEstimation_pipelines:
    def __init__(self,folder_out,cam_height_m=7,fov_x_deg=None,limit=100):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.cam_height_m_gt = cam_height_m
        self.fov_x_deg_gt = fov_x_deg
        self.limit = limit
        self.folder_out = folder_out
        self.V = utils_visualizer.visualizer(folder_out)
        self.VP = tools_vanishing.detector_VP(folder_out)
        self.TP = tools_time_profiler.Time_Profiler()
        self.folder_temp_frames = self.folder_out + 'frames' + '/'
        self.folder_temp_boxes  = self.folder_out + 'boxes'  + '/'

        return
# ----------------------------------------------------------------------------------------------------------------------
    def serialize_cam_params(self, vp_ver, vp_hor, fov_x_deg, fov_y_deg, cam_height):
        dct_res = {}
        dct_res['vp_ver_x'] = vp_ver[0]
        dct_res['vp_ver_y'] = vp_ver[1]
        dct_res['vp_hor_x'] = vp_hor[0]
        dct_res['vp_hor_y'] = vp_hor[1]
        dct_res['fov_x_deg'] = fov_x_deg
        dct_res['fov_y_deg'] = fov_y_deg
        dct_res['cam_height_pixels'] = cam_height
        return dct_res
# ----------------------------------------------------------------------------------------------------------------------
    def deserialize_cam_params(self,dct_res):
        vp_ver = (dct_res['vp_ver_x'], dct_res['vp_ver_y'])
        vp_hor = (dct_res['vp_hor_x'], dct_res['vp_hor_y'])
        fov_x_deg = dct_res['fov_x_deg']
        fov_y_deg = dct_res['fov_y_deg']
        cam_height = dct_res['cam_height_pixels']
        return vp_ver, vp_hor, fov_x_deg, fov_y_deg, cam_height
# ----------------------------------------------------------------------------------------------------------------------
    def stream_to_images(self,URL, folder_out, limit=100):
        folder_out = self.folder_out if folder_out is None else folder_out
        tools_IO.remove_files(folder_out,'*.jpg',create=True)
        if URL[-1]=='/':
            for file_name in tools_IO.get_filenames(URL,'*.jpg')[:limit]:
                copyfile(URL + '/' + file_name, folder_out + file_name)
        else:
            tools_video.extract_frames_v2(URL, folder_out, prefix='', start_frame=0, end_frame=limit, step=1, scale=1,silent=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pipeline_post_process_detection(self,df_boxes,size_min=20,size_max=900,th_min_iou=0.25):

        if df_boxes.shape[0] ==0:
            return df_boxes

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
                for box in df_prev.iloc[:,3:].values:iou = max(iou, self.VP.iou(box_cur, box))
                for box in df_next.iloc[:,3:].values:iou = max(iou, self.VP.iou(box_cur, box))
                if iou>th_min_iou:
                    res.append(df_cur.iloc[b,:].values)
                else:
                    res_bad.append(df_cur.iloc[b, :].values)

        df_res = pd.DataFrame(res,columns=df_boxes.columns)
        #df_res_bad = pd.DataFrame(res_bad,columns=df_boxes.columns)
        #print(df_boxes.shape[0],df_res_bad.shape[0])

        return df_res

# ----------------------------------------------------------------------------------------------------------------------
    def get_cam_params(self,filenames,image_clear_bg,df_boxes=None,do_debug=False):

        if image_clear_bg is None:
            image_clear_bg = cv2.imread(self.folder_temp_frames + filenames[0])

        image = cv2.imread(self.folder_temp_frames + filenames[0]) if image_clear_bg is None else image_clear_bg

        # method 1 - on background
        lines_ver = self.VP.get_lines_ver_candidates_single_image(image_clear_bg, do_debug=do_debug)
        #lines_ver = self.VP.get_lines_ver_candidates_static(self.folder_temp_frames, do_debug=do_debug)#lines_ver = U_VP.get_lines_ver_candidates_dynamic(folder_in,do_debug=True)
        #lines_ver = self.VP.get_lines_ver_candidates_static(self.folder_temp_frames, len_min=55, len_max=200,df_boxes=df_boxes, do_debug=do_debug)

        vp_ver, lines_vp_ver = self.VP.get_vp(self.VP.reshape_lines_as_paired(lines_ver), filename_out='VP_ver.png', image_debug=image)

        method = 2
        if method==1:# on background
            lines_hor = self.VP.get_lines_hor_candidates_single_image(image_clear_bg, do_debug=do_debug)
        elif method==2:# cars only
            lines_hor = self.VP.get_lines_hor_candidates_static(self.folder_temp_frames, df_boxes=df_boxes,do_debug=do_debug)
        else:# full frames
            lines_hor = self.VP.get_lines_hor_candidates_static(self.folder_temp_frames,do_debug=do_debug)

        vp_hor, lines_vp_hor = self.VP.get_vp(self.VP.reshape_lines_as_paired(lines_hor), filename_out='VP_hor.png', image_debug=image)
        focal_lenth = self.VP.get_focal_length([vp_ver, vp_hor])
        fov_x_deg = 2 * numpy.arctan(0.5 * self.V.W / focal_lenth) * 180 / numpy.pi
        fov_y_deg = fov_x_deg * image.shape[0] / image.shape[1]
        image_BEV_proxy, h_ipersp, cam_height, p_camera_BEV_xy, center_BEV,lines_edges = self.VP.build_BEV_by_fov_van_point(image, fov_x_deg, fov_y_deg,vp_ver,vp_hor, do_rotation=True)

        dct_res = self.serialize_cam_params(vp_ver, vp_hor, fov_x_deg, fov_y_deg, cam_height)
        return dct_res
# ----------------------------------------------------------------------------------------------------------------------
    def cleanup(self):
        tools_IO.remove_files(self.folder_temp_frames)
        tools_IO.remove_files(self.folder_temp_boxes)
        if os.path.isdir(self.folder_temp_frames):
            os.rmdir(self.folder_temp_frames)
        if os.path.isdir(self.folder_temp_frames):
            os.rmdir(self.folder_temp_frames)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def regress_fov(self, vp_ver, vp_hor, df_boxes_cars, cam_height_m_gt):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        empty = numpy.full((self.V.H,self.V.W,3),0,dtype=numpy.uint8)
        fov_x_degs = numpy.arange(6.0, 20.0, 1.0)
        loss = []
        for fov_x_deg in fov_x_degs:
            #print('regress', fov_x_deg)
            fov_y_deg = fov_x_deg*self.V.H / self.V.W
            df_objects = self.VP.prepare_cuboids_data(vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_cars)
            pix_per_meter_BEV = self.VP.evaluate_pix_per_meter_BEV_cars(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor)
            image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.VP.build_BEV_by_fov_van_point(empty, fov_x_deg, fov_y_deg,vp_ver,vp_hor, do_rotation=True)
            loss.append(abs((cam_height_px / pix_per_meter_BEV)-cam_height_m_gt))
            #print('cam_height_m',(cam_height_px / pix_per_meter_BEV))

        i =numpy.argmin(numpy.array(loss))
        fov_x_deg = fov_x_degs[i]
        fov_y_deg = fov_x_deg * self.V.H / self.V.W
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return fov_x_deg,fov_y_deg
# ----------------------------------------------------------------------------------------------------------------------
    def draw_BEVs_cars_folder_wrapper(self, df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, folder_temp_frames, image_clear_bg):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        self.V.draw_BEVs_cars_folder(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, folder_temp_frames, image_clear_bg=image_clear_bg)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def draw_BEVs_lps_folder_wrapper(self, df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, folder_temp_frames, image_clear_bg):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        self.V.draw_BEVs_lps_folder(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, folder_temp_frames)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def update_detectors(self,filename):
        image = cv2.imread(filename)
        self.V.H, self.V.W = image.shape[:2]
        self.VP.H, self.VP.W = image.shape[:2]
        self.V.VP.H, self.V.VP.W = image.shape[:2]
        return
# ----------------------------------------------------------------------------------------------------------------------
    def E2E_v1_cars(self, URL, df_boxes_cars=None, from_cache = True, do_debug=True):

        if not from_cache:
            tools_IO.remove_files(self.folder_out,create=True)
            tools_IO.remove_files(self.folder_temp_boxes)
            self.stream_to_images(URL, self.folder_temp_frames, limit=self.limit)

        filenames = tools_IO.get_filenames(self.folder_temp_frames, '*.jpg')
        if len(filenames) == 0: return
        self.update_detectors(self.folder_temp_frames + filenames[0])

        if df_boxes_cars is None:

            from detector import detector_TF_Zoo
            D = detector_TF_Zoo.detector_TF_Zoo(self.folder_temp_boxes)
            df_boxes_cars = D.process_folder(self.folder_temp_frames, do_debug=do_debug)
            df_boxes_cars = self.pipeline_post_process_detection(df_boxes_cars)
            df_boxes_cars.to_csv(self.folder_out + 'df_boxes_filtered.csv', index=False)

        if from_cache and os.path.isfile(self.folder_out + 'result.json'):
            image_clear_bg = cv2.imread(self.folder_out + 'background_clean.png')
            dct_res = tools_IO.load_dict(self.folder_out + 'result.json')
        else:
            image_clear_bg = self.V.remove_bg(df_boxes_cars, self.folder_temp_frames)
            cv2.imwrite(self.folder_out + 'background_clean.png', image_clear_bg)
            dct_res = self.get_cam_params(filenames, image_clear_bg, df_boxes_cars)

        vp_ver, vp_hor, fov_x_deg, fov_y_deg, cam_height = self.deserialize_cam_params(dct_res)
        df_boxes_cars = tools_DF.apply_filter(df_boxes_cars, 'ID', tools_IO.get_filenames(self.folder_temp_frames, '*.jpg'))

        if from_cache and os.path.isfile(self.folder_out + 'df_objects.csv'):
            df_objects = pd.read_csv(self.folder_out + 'df_objects.csv')
        else:
            df_objects = None

        if self.fov_x_deg_gt is not None:
            # quick
            fov_x_deg = self.fov_x_deg_gt
            fov_y_deg = fov_x_deg * self.V.H / self.V.W
            if df_objects is None:
                df_objects = self.VP.prepare_cuboids_data(vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_cars)
            pix_per_meter_BEV = self.VP.evaluate_pix_per_meter_BEV_cars(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor)
        elif self.cam_height_m_gt is not None:
            # slow
            fov_x_deg, fov_y_deg = self.regress_fov(vp_ver, vp_hor, df_boxes_cars, self.cam_height_m_gt)
            image_BEV, h_ipersp, cam_height_px, p_camera_BEV_xy, p_center_BEV_xy, lines_edges = self.VP.build_BEV_by_fov_van_point(image_clear_bg, fov_x_deg, fov_y_deg,vp_ver,vp_hor, do_rotation=True)
            if df_objects is None:
                df_objects = self.VP.prepare_cuboids_data(vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_cars)
            pix_per_meter_BEV = cam_height_px / self.cam_height_m_gt
        else:
            if df_objects is None:
                df_objects = self.VP.prepare_cuboids_data(vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_cars)
            pix_per_meter_BEV = self.VP.evaluate_pix_per_meter_BEV_cars(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor)

        dct_res['fov_x_deg']=fov_x_deg
        dct_res['fov_y_deg']=fov_y_deg
        dct_res['stream'] = URL
        tools_IO.save_dict(dct_res, filename_out=self.folder_out+'result.json')
        df_objects.to_csv(self.folder_out+'df_objects.csv',index=False)

        self.draw_BEVs_cars_folder_wrapper(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, self.folder_temp_frames, image_clear_bg = image_clear_bg)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def E2E_v2_LPs(self, URL, df_boxes_lps, from_cache = False):

        if not from_cache:
            tools_IO.remove_files(self.folder_out, create=True)
            tools_IO.remove_files(self.folder_temp_boxes)
            self.stream_to_images(URL, self.folder_temp_frames, limit=self.limit)

        filenames = tools_IO.get_filenames(self.folder_temp_frames, '*.jpg')
        if len(filenames) == 0: return
        self.update_detectors(self.folder_temp_frames + filenames[0])
        image_clear_bg = None
        if from_cache:
            dct_res = tools_IO.load_dict(self.folder_out + 'result.json')
        else:
            dct_res = self.get_cam_params(filenames, image_clear_bg=None, df_boxes=None)

        vp_ver, vp_hor, fov_x_deg, fov_y_deg, cam_height = self.deserialize_cam_params(dct_res)
        df_boxes_lps = tools_DF.apply_filter(df_boxes_lps, 'ID', tools_IO.get_filenames(self.folder_temp_frames, '*.jpg'))

        df_objects = self.VP.prepare_angles_data(vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_lps)
        pix_per_meter_BEV = self.VP.evaluate_pix_per_meter_BEV_lps(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor)

        dct_res['fov_x_deg'] = fov_x_deg
        dct_res['fov_y_deg'] = fov_y_deg
        dct_res['stream'] = URL
        tools_IO.save_dict(dct_res, filename_out=self.folder_out + 'result.json')
        self.draw_BEVs_lps_folder_wrapper(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, self.folder_temp_frames, image_clear_bg=image_clear_bg)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def bruteforce_fov(self, URL, df_boxes_cars=None, from_cache=False, do_debug=True):

        if not from_cache:
            tools_IO.remove_files(self.folder_out,create=True)
            tools_IO.remove_files(self.folder_temp_boxes)
            self.stream_to_images(URL, self.folder_temp_frames, limit=self.limit)

        filenames = tools_IO.get_filenames(self.folder_temp_frames, '*.jpg')
        if len(filenames) == 0: return
        self.update_detectors(self.folder_temp_frames + filenames[0])

        if df_boxes_cars is None:
            from detector import detector_TF_Zoo
            D = detector_TF_Zoo.detector_TF_Zoo(self.folder_temp_boxes)
            df_boxes_cars = D.process_folder(self.folder_temp_frames, do_debug=do_debug)
            df_boxes_cars = self.pipeline_post_process_detection(df_boxes_cars)
            df_boxes_cars.to_csv(self.folder_out + 'df_boxes_filtered.csv', index=False)

        if from_cache:
            image_clear_bg = cv2.imread(self.folder_out + 'background_clean.png')
            dct_res = tools_IO.load_dict(self.folder_out + 'result.json')
        else:
            image_clear_bg = self.V.remove_bg(df_boxes_cars, self.folder_temp_frames)
            cv2.imwrite(self.folder_out + 'background_clean.png', image_clear_bg)
            dct_res = self.get_cam_params(filenames, image_clear_bg, df_boxes_cars)
        vp_ver, vp_hor, fov_x_deg, fov_y_deg, cam_height = self.deserialize_cam_params(dct_res)

        df_boxes_cars = tools_DF.apply_filter(df_boxes_cars, 'ID', tools_IO.get_filenames(self.folder_temp_frames, '*.jpg'))
        df_cnts = tools_DF.my_agg(df_boxes_cars, cols_groupby=['ID'], cols_value=['score'], aggs=['count'], list_res_names=['#'])
        filename = df_cnts.sort_values(by='#',ascending=False).iloc[0,0]
        fov_x_degs = numpy.arange(34.0, 55.0, 1)
        for fov_x_deg in fov_x_degs:
            print(('bruteforce_fov: %.1f'+u'\u00B0')%fov_x_deg)
            fov_y_deg = fov_x_deg * self.V.H / self.V.W

            df_objects = self.VP.prepare_cuboids_data(vp_ver, vp_hor, fov_x_deg, fov_y_deg, df_boxes_cars)
            pix_per_meter_BEV = self.VP.evaluate_pix_per_meter_BEV_cars(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor)
            self.V.draw_BEVs_cars_folder(df_objects, fov_x_deg, fov_y_deg, vp_ver, vp_hor, pix_per_meter_BEV, self.folder_temp_frames, image_clear_bg=image_clear_bg, list_of_masks=filename)

        return
# ----------------------------------------------------------------------------------------------------------------------
