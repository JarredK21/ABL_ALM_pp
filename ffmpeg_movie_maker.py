#working on local computer
import ffmpeg
import os
import operator
import sys


#directories
in_dir = "ISOplots/"

video_folder = in_dir + "videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)


plot_l = True; plot_r = True; plot_tr = True; plot_i = True; plot_t = True
planes_plot = [plot_l,plot_r,plot_tr,plot_i,plot_t]

#check if no velocity components selected
if all(list(map(operator.not_, planes_plot))) == True:
    sys.exit("error no velocity component selected")


#loop over true planes
planes = ["l","r", "tr","i","t"]
plane_labels = ["horizontal","rotor", "transverse rotor", "inflow", "longitudinal"]
ip = 0
for plane in planes:
    if planes_plot[ip] == False:
        ip+=1
        continue

    if plane == "l":
        offsets = [22.5,85,142.5]
    elif plane == "r":
        offsets = [-5.5,-63.0]
    elif plane == "tr":
        offsets = [0.0]
    elif plane == "i":
        offsets = [0.0]
    elif plane == "t":
        offsets = [0.0]

    ic = 0
    for offset in offsets:

        plot_u = False; plot_v = False; plot_w = True; plot_hvelmag = True
        velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]

        #check if no velocity components selected
        if all(list(map(operator.not_, velocity_plot))) == True:
            sys.exit("error no velocity component selected")
        
        
        #loop over true velocity components
        velocity_comps = ["velocityx","velocityy","velocityz","Horizontal_velocity"]
        iv = 0
        for velocity_comp in velocity_comps:
            if velocity_plot[iv] == False:
                iv+=1
                continue
            
            print(plane_labels[ip],velocity_comps[iv],offset)

            plots_dir = in_dir+"{}_{}_{}/".format(plane_labels[ip],velocity_comp,offset)
            filename = "{}_{}_{}/".format(plane_labels[ip],velocity_comp,offset)



            FRAMERATE = 4
            (
                ffmpeg
                .input(plots_dir+"*.png", pattern_type="glob", framerate=FRAMERATE)
                .output(video_folder+"{}.mp4".format(filename))
                .run()
            )

            print(plane_labels[ip],velocity_comps[iv],offset)

            iv+=1
        ic+=1
    ip+=1