from utils.tracklet_tools import read_objects
import os
# os.environ['DISPLAY'] = ':0'
from collections import defaultdict
from matplotlib import pyplot as plt


# a test case
os.makedirs('./test_output/', exist_ok=True)

objects = read_objects('./cmax01_corrected.xml')
tzs = defaultdict(list)
for obs in objects:
    # get the car's content:
    # for frame
    for i, pose in enumerate(obs):
        tzs[i].append(pose[1][2])
    pass

# for one_frame_objs in objects[0]:
#     for obj in one_frame_objs:
#         tzs[obj[0]].append(obj[1][2])

frame_ids = []
tz_v = []
for k, v in tzs.items():
    frame_ids.append(k)
    tz_v.append(v)

plt.plot(frame_ids, tz_v)
print(tzs)