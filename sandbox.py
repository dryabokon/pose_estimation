import cv2
import numpy
# ----------------------------------------------------------------------------------------------------------------------
import tools_animation
import tools_video
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
#folder_in = './images/ex_MELCO/'
folder_in = './images/ex_xxx/'
# ----------------------------------------------------------------------------------------------------------------------
vp_ver = numpy.array([1560.23, -446.74])
vp_hor = numpy.array([-29043.53,234.27])
fov_x_deg = 14.6
fov_y_deg = fov_x_deg*1080/1920.0#8.21
pix_per_meter_BEV = 53/3.0
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tools_animation.folder_to_video(folder_in, folder_out+'KZ_small.mp4', mask='*.jpg', framerate=24)