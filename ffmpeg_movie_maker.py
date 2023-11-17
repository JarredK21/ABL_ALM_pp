import os
import glob
import re
import ffmpeg_movie_maker
import ffmpeg

#whether or not folder exists execute code
#sort files
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


folder = "../../test/rotor_Plane_Total_Horizontal_velocity_0.0/"

    
#sort files
files = glob.glob(folder+"*.png")
files.sort(key=natural_keys)

it = 1
for file in files:
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)
    os.rename(file,folder+"rotor_Plane_Total_Horizontal_velocity_0.0_{}.png".format(Time_idx))
    it+=1


FRAMERATE = 4
(
    ffmpeg
    .input(folder+"*.png", pattern_type="glob", framerate=FRAMERATE)
    .output(folder+"movie.mp4")
    .run()
)