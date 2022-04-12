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
from CV import tools_alg_match
from CV import tools_Skeletone
# ---------------------------------------------------------------------------------------------------------------------
class detector_Vehicles:
    def __init__(self, folder_out):
        self.folder_out = folder_out
        self.Ske = tools_Skeletone.Skelenonizer(folder_out=folder_out)
        self.P = tools_plot_v2.Plotter(folder_out=folder_out)

        self.TP = tools_time_profiler.Time_Profiler()
        self.H = 1080
        self.W = 1920
        self.step = 10
        return
# ---------------------------------------------------------------------------------------------------------------------
    def xxx(self,folder_in):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        filenames = tools_IO.get_filenames(folder_in, '*.jpg')
        img_cur = tools_image.desaturate_2d(cv2.imread(folder_in + filenames[0]))

        for filename_cur in filenames[1:200]:
            img_prev = img_cur.copy()
            img_cur = tools_image.desaturate_2d(cv2.imread(folder_in + filename_cur))
            flow = cv2.calcOpticalFlowFarneback(img_prev, img_cur, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            y, x = numpy.mgrid[self.step // 2:self.H:self.step, self.step // 2:self.W:self.step].reshape(2, -1)
            fx, fy = flow[y, x].T
            lines = numpy.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = numpy.int32(lines + 0.5)
            result = cv2.cvtColor(next, cv2.COLOR_GRAY2BGR)
            cv2.polylines(result, lines, 0, (0, 255, 0))
            for (x1, y1), (x2, y2) in lines:
                cv2.circle(result, (x1, y1), 1, (0, 255, 0), -1)

            # th = 4
            # y, x = np.mgrid[0:h:1, 0:w:1].reshape(2, -1)
            # fx, fy = flow[y, x].T
            # fx = fx.reshape((h,w))
            # fy = fy.reshape((h, w))
            # mask  = 255*(( (fx>th) + (fy>th) ) > 0 )
            # #result = mask
            # result = tools_image.put_color_by_mask(frame,mask,(0,0,200))

        return
