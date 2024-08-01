# blender -P visualize.py -- --render_engine eevee\
#                         --bvh_list  ./bvh_files/greeting/Standing_Greeting_Ortiz.bvh ./bvh_files/greeting/Standing_Greeting_Mousey.bvh \
#                         ./bvh_files/greeting/Mousey_Standing_Greeting_SAN.bvh ./bvh_files/greeting/Mousey_Greeting_ours1.bvh\
#                         ./bvh_files/greeting/r2et_Standing_Greeting.bvh \
#                         --fbx_list ./skin/Ortiz.fbx ./skin/Mousey.fbx ./skin/Mousey.fbx ./skin/Mousey.fbx ./skin/Mousey.fbx\
#                         --frame_end 60 --fps 30
# blender -P visualize.py -- --render_engine eevee\
#                         --bvh_list  ./bvh_files/baseball_pitching/Baseball_Pitching.bvh ./bvh_files/baseball_pitching/Kaya_Baseball_Pitching_copy.bvh  \
#                         ./bvh_files/baseball_pitching/Kaya_Baseball_Pitching_SAN.bvh ./bvh_files/baseball_pitching/Kaya_Baseball_Pitching_R2ET.bvh\
#                         ./bvh_files/baseball_pitching/Kaya_Baseball_Pitching_ours.bvh \
#                         --fbx_list ./skin/Ybot.fbx ./skin/Kaya.fbx ./skin/Kaya.fbx  ./skin/Kaya.fbx ./skin/Kaya.fbx\
#                         --frame_end 119 --fps 30
# blender -P visualize.py -- --render_engine eevee\
#                         --bvh_list  ./bvh_files/thankful/Ortiz_Thankful.bvh \
#                         ./bvh_files/thankful/Ortiz_Thankful_geo.bvh ./bvh_files/thankful/Ortiz_Thankful_ours.bvh\
#                         ./bvh_files/thankful/Ortiz_Thankful_ours1.bvh ./bvh_files/thankful/Ortiz_Thankful_ours1.bvh\
#                         --fbx_list ./skin/Ortiz.fbx ./skin/Ortiz.fbx ./skin/Ortiz.fbx ./skin/Ortiz.fbx ./skin/Ortiz.fbx\
#                         --frame_end 60 --fps 30
blender -P visualize.py -- --render_engine eevee\
                        --bvh_list  ./bvh_files/clapping/Clapping.bvh ./bvh_files/clapping/Clapping_AJ.bvh ./bvh_files/clapping/Clapping_Kaya.bvh ./bvh_files/clapping/Clapping_mousey.bvh\
                        --fbx_list ./skin/Ybot.fbx ./skin/Aj.fbx ./skin/Kaya.fbx ./skin/Mousey.fbx\
                        --frame_end 60 --fps 30