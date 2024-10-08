#working on local computer
import ffmpeg
import os
import operator
import sys


#directories
in_dir = "../../NREL_5MW_MCBL_E_CRPM/post_processing/deforming_blade/"

video_folder = in_dir+"videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)

filenames = ["LPF_1_traj_200_260","LPF_1_traj_720_760","LPF_1_traj_760_820",
             "LPF_2_200_260","LPF_2_720_760","LPF_2_760_820","HPF_1_200_260","HPF_1_720_760","HPF_1_760_820"]
for filename in filenames:
    plots_dir = in_dir+filename

    FRAMERATE = 4
    (
        ffmpeg
        .input(plots_dir+"/*.png", pattern_type="glob", framerate=FRAMERATE)
        .output(video_folder+"{}.mp4".format(filename))
        .run()
    )


# plot_l = False; plot_r = False; plot_tr = False; plot_i = False; plot_t = True
# planes_plot = [plot_l,plot_r,plot_tr,plot_i,plot_t]

# #check if no velocity components selected
# if all(list(map(operator.not_, planes_plot))) == True:
#     sys.exit("error no velocity component selected")


# #loop over true planes
# planes = ["l","r", "tr","i","t"]
# plane_labels = ["horizontal","rotor", "transverse rotor", "inflow", "longitudinal"]
# ip = 0
# for plane in planes:
#     if planes_plot[ip] == False:
#         ip+=1
#         continue

#     if plane == "l":
#         offsets = [85]
#     elif plane == "r":
#         offsets = [-5.5]
#     elif plane == "tr":
#         offsets = [0.0]
#     elif plane == "i":
#         offsets = [0.0]
#     elif plane == "t":
#         offsets = [0.0]

#     ic = 0
#     for offset in offsets:

#         plot_u = False; plot_v = False; plot_w = True; plot_hvelmag = True
#         velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]
#         Fluctuating_velocity = False

#         #check if no velocity components selected
#         if all(list(map(operator.not_, velocity_plot))) == True:
#             sys.exit("error no velocity component selected")
        
        
#         #loop over true velocity components
#         velocity_comps = ["velocityx","velocityy","velocityz","Horizontal_velocity"]
#         iv = 0
#         for velocity_comp in velocity_comps:
#             if velocity_plot[iv] == False:
#                 iv+=1
#                 continue
            
#             print(plane_labels[ip],velocity_comps[iv],offset)

#             if Fluctuating_velocity == True:
#                 plots_dir = in_dir+"{}_Plane_Fluctutating_{}_{}/".format(plane_labels[ip],velocity_comp,offset)
#                 filename = "{}_Fluc_{}_{}".format(plane_labels[ip],velocity_comp,offset)
#             else:
#                 plots_dir = in_dir+"{}_Plane_Total_{}_{}_2/".format(plane_labels[ip],velocity_comp,offset)
#                 filename = "{}_{}_{}".format(plane_labels[ip],velocity_comp,offset)

#             FRAMERATE = 4
#             (
#                 ffmpeg
#                 .input(plots_dir+"*.png", pattern_type="glob", framerate=FRAMERATE)
#                 .output(video_folder+"{}.mp4".format(filename))
#                 .run()
#             )

#             print(plane_labels[ip],velocity_comps[iv],offset)

#             iv+=1
#         ic+=1
#     ip+=1