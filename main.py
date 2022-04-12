import argparse
# ----------------------------------------------------------------------------------------------------------------------
import pipelines
# ----------------------------------------------------------------------------------------------------------------------
input_aset = '_KZ.mp4'
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to stream or video e.g.(rtsp://localhost:8554/live.stream)', default=input_aset)
    parser.add_argument('--output', '-out', help='path to output folder',default=folder_out)
    args = parser.parse_args()
    Pipe = pipelines.PoseEstimation_pipelines(args.output)
    Pipe.evaluate_VP_FOV_height(args.input,do_debug=True)




