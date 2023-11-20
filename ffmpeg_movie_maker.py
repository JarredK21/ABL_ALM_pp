import ffmpeg
import os

folder = "../../test/rotor_Plane_Total_Horizontal_velocity_0.0/"

video_folder = folder + "videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)

FRAMERATE = 4
(
    ffmpeg
    .input(folder+"*.png", pattern_type="glob", framerate=FRAMERATE)
    .output(folder+"movie.mp4")
    .run()
)