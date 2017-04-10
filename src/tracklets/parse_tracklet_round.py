""" Tracklet XML file parsing
This code was taken as is from the kitti website link
(http://cvlibs.net/datasets/kitti/downloads/parseTrackletXML.py ).
Minor Pythonic naming changes made and TRUNC_UNSET enum addition.
Original header and author comments follow below.

---
parse XML files containing tracklet info for kitti data base (raw data section)
(http://cvlibs.net/datasets/kitti/raw_data.php)

No guarantees that this code is correct, usage is at your own risk!

created by Christian Herdtweck, Max Planck Institute for Biological Cybernetics
  (christian.herdtweck@tuebingen.mpg.de)

requires numpy!
"""

# Version History:
# 4/7/12 Christian Herdtweck: seems to work with a few random test xml tracklet files;
#   converts file contents to ElementTree and then to list of Tracklet objects;
#   Tracklet objects have str and iter functions
# 5/7/12 ch: added constants for state, occlusion, truncation and added consistency checks
# 30/1/14 ch: create example function from example code

from __future__ import print_function
from xml.etree.ElementTree import ElementTree
import numpy as np
import itertools
from warnings import warn

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0': STATE_UNSET, '1': STATE_INTERP, '2': STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1': OCC_UNSET, '0': OCC_VISIBLE, '1': OCC_PARTLY, '2': OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {
    '-1': TRUNC_UNSET,  # FIXME RW: Added this
    '99': TRUNC_UNSET,  # FIXME RW: Original code had this but 99 is supposed be 'behind'???
    '0': TRUNC_IN_IMAGE,
    '1': TRUNC_TRUNCATED,
    '2': TRUNC_OUT_IMAGE,
    '3': TRUNC_BEHIND_IMAGE}


class Tracklet(object):
    r""" representation an annotated object track

    Tracklets are created in function parseXML and can most conveniently used as follows:

    for trackletObj in parseXML(trackletFile):
      for translation, rotation, state, occlusion, truncation, amtOcclusion, amt_borders, absoluteFrameNumber in trackletObj:
        ... your code here ...
      #end: for all frames
    #end: for all tracklets

    absoluteFrameNumber is in range [first_frame, first_frame+num_frames[
    amtOcclusion and amt_borders could be None

    You can of course also directly access the fields objType (string), size (len-3 ndarray), first_frame/num_frames (int),
      trans/rots (num_frames x 3 float ndarrays), states/truncs (len-num_frames uint8 ndarrays), occs (num_frames x 2 uint8 ndarray),
      and for some tracklets amt_occs (num_frames x 2 float ndarray) and amt_borders (num_frames x 3 float ndarray). The last two
      can be None if the xml file did not include these fields in poses
    """

    object_type = None
    size = None  # len-3 float array: (height, width, length)
    first_frame = None
    trans = None  # n x 3 float array (x,y,z)
    rots = None  # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None  # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    amt_occs = None  # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amt_borders = None  # None (n x 3) float array  (amt_border_l / _r / _kf)
    num_frames = None

    def __init__(self):
        r""" create Tracklet with no info set """
        self.size = np.nan * np.ones(3, dtype=float)

    def __str__(self):
        r""" return human-readable string representation of tracklet object

        called implicitly in
        print trackletObj
        or in
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(self.num_frames, self.object_type)

    def __iter__(self):
        r""" returns an iterator that yields tuple of all the available data for each frame

        called whenever code iterates over a tracklet object, e.g. in
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amt_borders, absoluteFrameNumber in trackletObj:
          ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amt_occs is None:
            return itertools.izip(
                self.trans, self.rots, self.states, self.occs, self.truncs,
                itertools.repeat(None), itertools.repeat(None),
                range(self.first_frame, self.first_frame + self.num_frames))
        else:
            return itertools.izip(
                self.trans, self.rots, self.states, self.occs, self.truncs,
                self.amt_occs, self.amt_borders, range(self.first_frame, self.first_frame + self.num_frames))

# end: class Tracklet


def parse_xml(tracklet_file):
    r""" parse tracklet xml file and convert results to list of Tracklet objects

    :param tracklet_file: name of a tracklet xml file
    :returns: list of Tracklet objects read from xml file
    """

    # convert tracklet XML data to a tree structure
    etree = ElementTree()
    print('Parsing Tracklet file', tracklet_file)
    with open(tracklet_file) as f:
        etree.parse(f)

    # now convert output to list of Tracklet objects
    tracklets_elem = etree.find('tracklets')
    tracklets = []
    tracklet_idx = 0
    num_tracklets = None
    for tracklet_elem in tracklets_elem:
        if tracklet_elem.tag == 'count':
            num_tracklets = int(tracklet_elem.text)
            print('File contains', num_tracklets, 'Tracklets')
        elif tracklet_elem.tag == 'item_version':
            pass
        elif tracklet_elem.tag == 'item':
            new_track = Tracklet()
            is_finished = False
            has_amt = False
            frame_idx = None
            for info in tracklet_elem:
                if is_finished:
                    raise ValueError('More info on element after finished!')
                if info.tag == 'objectType':
                    new_track.object_type = info.text
                elif info.tag == 'h':
                    new_track.size[0] = round(float(info.text), 3)
                elif info.tag == 'w':
                    new_track.size[1] = round(float(info.text), 3)
                elif info.tag == 'l':
                    new_track.size[2] = round(float(info.text), 3)
                elif info.tag == 'first_frame':
                    new_track.first_frame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        if pose.tag == 'count':  # this should come before the others
                            if new_track.num_frames is not None:
                                raise ValueError('There are several pose lists for a single track!')
                            elif frame_idx is not None:
                                raise ValueError('?!')
                            new_track.num_frames = int(pose.text)
                            new_track.trans = np.nan * np.ones((new_track.num_frames, 3), dtype=float)
                            new_track.rots = np.nan * np.ones((new_track.num_frames, 3), dtype=float)
                            new_track.states = np.nan * np.ones(new_track.num_frames, dtype='uint8')
                            new_track.occs = np.nan * np.ones((new_track.num_frames, 2), dtype='uint8')
                            new_track.truncs = np.nan * np.ones(new_track.num_frames, dtype='uint8')
                            new_track.amt_occs = np.nan * np.ones((new_track.num_frames, 2), dtype=float)
                            new_track.amt_borders = np.nan * np.ones((new_track.num_frames, 3), dtype=float)
                            frame_idx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frame_idx is None:
                                raise ValueError('Pose item came before number of poses!')
                            for poseInfo in pose:
                                if poseInfo.tag == 'tx':
                                    new_track.trans[frame_idx, 0] = round(float(poseInfo.text), 3)
                                elif poseInfo.tag == 'ty':
                                    new_track.trans[frame_idx, 1] = round(float(poseInfo.text), 3)
                                elif poseInfo.tag == 'tz':
                                    new_track.trans[frame_idx, 2] = round(float(poseInfo.text), 3)
                                elif poseInfo.tag == 'rx':
                                    new_track.rots[frame_idx, 0] = round(float(poseInfo.text), 3)
                                elif poseInfo.tag == 'ry':
                                    new_track.rots[frame_idx, 1] = round(float(poseInfo.text), 3)
                                elif poseInfo.tag == 'rz':
                                    new_track.rots[frame_idx, 2] = round(float(poseInfo.text), 3)
                                elif poseInfo.tag == 'state':
                                    new_track.states[frame_idx] = stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    new_track.occs[frame_idx, 0] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    new_track.occs[frame_idx, 1] = occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    new_track.truncs[frame_idx] = truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    new_track.amt_occs[frame_idx, 0] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    new_track.amt_occs[frame_idx, 1] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    new_track.amt_borders[frame_idx, 0] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    new_track.amt_borders[frame_idx, 1] = float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    new_track.amt_borders[frame_idx, 2] = float(poseInfo.text)
                                    has_amt = True
                                else:
                                    raise ValueError('Unexpected tag in poses item: {0}!'.format(poseInfo.tag))
                            frame_idx += 1
                        else:
                            raise ValueError('Unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    is_finished = True
                else:
                    raise ValueError('Unexpected tag in tracklets: {0}!'.format(info.tag))
            # end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not is_finished:
                warn('Tracklet {0} was not finished!'.format(tracklet_idx))
            if new_track.num_frames is None:
                warn('Tracklet {0} contains no information!'.format(tracklet_idx))
            elif frame_idx != new_track.num_frames:
                warn('Tracklet {0} is supposed to have {1} frames, but parser found {1}!'.format(
                    tracklet_idx, new_track.num_frames, frame_idx))
            if np.abs(new_track.rots[:, :2]).sum() > 1e-16:
                warn('Track contains rotation other than yaw!')

            # if amt_occs / amt_borders are not set, set them to None
            if not has_amt:
                new_track.amt_occs = None
                new_track.amt_borders = None

            # add new tracklet to list
            tracklets.append(new_track)
            tracklet_idx += 1

        else:
            raise ValueError('Unexpected tracklet info')
    # end: for tracklet list items

    print('Loaded', tracklet_idx, 'Tracklets')

    # final consistency check
    if tracklet_idx != num_tracklets:
        warn('According to xml information the file has {0} tracklets, but parser found {1}!'.format(
            num_tracklets, tracklet_idx))
    return tracklets
