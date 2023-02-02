import argparse
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
import pipelines
# ----------------------------------------------------------------------------------------------------------------------
def cars_to_LPs(df_boxes):

    ratio_x = 3.5
    ratio_y = 0.35

    res = []
    for r in range(df_boxes.shape[0]):
        x1 = df_boxes.iloc[r]['x1']
        y1 = df_boxes.iloc[r]['y1']
        x2 = df_boxes.iloc[r]['x2']
        y2 = df_boxes.iloc[r]['y2']
        x1,x2 = (x1+x2)/2-0.5*(x2-x1)/ratio_x,(x1+x2)/2+0.5*(x2-x1)/ratio_x
        h = (x2-x1)*112/520
        y1,y2 = y1*(ratio_y) + y2*(1-ratio_y) - 0.5*h,y1*(ratio_y) + y2*(1-ratio_y) + 0.5*h
        res.append([df_boxes.iloc[r]['ID'],df_boxes.iloc[r]['score'],df_boxes.iloc[r]['class'],x1,y1,x2,y2])

    df_boxes_lps = pd.DataFrame(res,columns=['ID','score','class','x1','y1','x2','y2'])

    return df_boxes_lps
# ----------------------------------------------------------------------------------------------------------------------
df_boxes = None
cam_height_m = None
cam_fov_x_deg = None
URL = None
from_cache = False
# ----------------------------------------------------------------------------------------------------------------------
# cam_height_m,cam_fov_x_deg,df_boxes=None,None,None
# URL = './KZ_small.mp4'
# ----------------------------------------------------------------------------------------------------------------------
# URL = './KZ_small.mp4'
# df_boxes_cars = pd.read_csv('./images/ex_KZ2/df_boxes_filtered.csv')
# cam_fov_x_deg=14
# ----------------------------------------------------------------------------------------------------------------------
URL = './images/ex_CZ/TNO-7180R_20220418134537.avi'
df_boxes_cars = pd.read_csv('./images/ex_CZ/df_boxes_filtered_TNO_7180R_20220418134537.csv')
#df_boxes_lps = cars_to_LPs(df_boxes_cars)
cam_height_m=7.8
cam_fov_x_deg = 8.5
# ----------------------------------------------------------------------------------------------------------------------
# URL = './images/ex_CZ/TNO-7180R_20220418135426.avi'
# df_boxes = pd.read_csv('./images/ex_CZ/df_boxes_filtered_TNO_7180R_20220418135426.csv')[:118]
# cam_fov_x_deg=7
# ----------------------------------------------------------------------------------------------------------------------
# URL = './images/ex_CZ/TNO-7180R_20220525174045.avi'
# df_boxes = pd.read_csv('./images/ex_CZ/df_boxes_filtered_TNO-7180R_20220525174045.csv')
# cam_fov_x_deg=27
# #cam_height_m=7.8
# from_cache = True
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './output/'
limit = 150
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to stream or video e.g.(rtsp://localhost:8554/live.stream)', default=URL)
    parser.add_argument('--output', '-out', help='path to output folder',default=folder_out)
    parser.add_argument('--debug', help='do debug True/False',type=bool,default=True)
    parser.add_argument('--limit', help='# of frames from the input', default=limit)
    parser.add_argument('--cam_height_m', help='cam height in meters', type=float,default=cam_height_m)
    parser.add_argument('--cam_fov_x_deg', help='cam fov x degrees', type=float,default=cam_fov_x_deg)
    args = parser.parse_args()

    Pipe = pipelines.PoseEstimation_pipelines(folder_out=args.output,cam_height_m=args.cam_height_m,fov_x_deg=args.cam_fov_x_deg,limit=args.limit)

    if args.cam_height_m is None and args.cam_fov_x_deg is None:
        Pipe.bruteforce_fov(URL=args.input, df_boxes_cars=df_boxes_cars, from_cache=from_cache, do_debug=args.debug)
    else:
        #Pipe.E2E_v2_LPs(URL=args.input, df_boxes_lps=df_boxes_lps)
        Pipe.E2E_v1_cars(URL=args.input, df_boxes_cars=df_boxes_cars, do_debug=args.debug)



