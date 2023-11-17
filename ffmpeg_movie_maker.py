import ffmpeg


folder = "../../test/rotor_Plane_Total_Horizontal_velocity_0.0/"

FRAMERATE = 4
(
    ffmpeg
    .input(folder+"*.png", pattern_type="glob", framerate=FRAMERATE)
    .output(folder+"movie.mp4")
    .run()
)