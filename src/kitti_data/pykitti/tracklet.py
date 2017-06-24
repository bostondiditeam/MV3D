#!/usr/bin/env python
"""
parse XML files containing tracklet info for kitti data base (raw data section)
(http://cvlibs.net/datasets/kitti/raw_data.php)

No guarantees that this code is correct, usage is at your own risk!

created by Christian Herdtweck, Max Planck Institute for Biological Cybernetics
  (christian.herdtweck@tuebingen.mpg.de)

requires numpy!

example usage:
  import parseTrackletXML as xmlParser
  kittiDir = '/path/to/kitti/data'
  drive = '2011_09_26_drive_0001'
  xmlParser.example(kittiDir, drive)
or simply on command line:
  python parseTrackletXML.py
"""

# Version History:
# 4/7/12 Christian Herdtweck: seems to work with a few random test xml tracklet files;
#   converts file contents to ElementTree and then to list of Tracklet objects;
#   Tracklet objects have str and iter functions
# 5/7/12 ch: added constants for state, occlusion, truncation and added consistency checks
# 30/1/14 ch: create example function from example code

from sys import argv as cmdLineArgs
from xml.etree.ElementTree import ElementTree
import numpy as np
import itertools
from warnings import warn
from config import cfg

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0':STATE_UNSET, '1':STATE_INTERP, '2':STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1':OCC_UNSET, '0':OCC_VISIBLE, '1':OCC_PARTLY, '2':OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99':TRUNC_UNSET, '0':TRUNC_IN_IMAGE, '1':TRUNC_TRUNCATED, \
                  '2':TRUNC_OUT_IMAGE, '3': TRUNC_BEHIND_IMAGE}


class Tracklet(object):
  r""" representation an annotated object track

  Tracklets are created in function parseXML and can most conveniently used as follows:

  for trackletObj in parseXML(trackletFile):
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
      ... your code here ...
    #end: for all frames
  #end: for all tracklets

  absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
  amtOcclusion and amtBorders could be None

  You can of course also directly access the fields objType (string), size (len-3 ndarray), firstFrame/nFrames (int),
    trans/rots (nFrames x 3 float ndarrays), states/truncs (len-nFrames uint8 ndarrays), occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amtOccs (nFrames x 2 float ndarray) and amtBorders (nFrames x 3 float ndarray). The last two
    can be None if the xml file did not include these fields in poses
  """

  objectType = None
  size = None  # len-3 float array: (height, width, length)
  firstFrame = None
  trans = None   # n x 3 float array (x,y,z)
  rots = None    # n x 3 float array (x,y,z)
  states = None  # len-n uint8 array of states
  occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
  truncs = None  # len-n uint8 array of truncation
  amtOccs = None    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
  amtBorders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
  nFrames = None

  def __init__(self):
    r""" create Tracklet with no info set """
    self.size = np.nan*np.ones(3, dtype=float)

  def __str__(self):
    r""" return human-readable string representation of tracklet object

    called implicitly in
    print trackletObj
    or in
    text = str(trackletObj)
    """
    return '[Tracklet over {0} frames for {1}]'.format(self.nFrames, self.objectType)

  def __iter__(self):
    r""" returns an iterator that yields tuple of all the available data for each frame

    called whenever code iterates over a tracklet object, e.g. in
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:
      ...do something ...
    or
    trackDataIter = iter(trackletObj)
    """
    if self.amtOccs is None:
      return zip(self.trans, self.rots, self.states, self.occs, self.truncs, \
          itertools.repeat(None), itertools.repeat(None), range(self.firstFrame, self.firstFrame+self.nFrames))
    else:
      return zip(self.trans, self.rots, self.states, self.occs, self.truncs, \
          self.amtOccs, self.amtBorders, range(self.firstFrame, self.firstFrame+self.nFrames))
#end: class Tracklet


def parseXML(trackletFile):
  r""" parse tracklet xml file and convert results to list of Tracklet objects

  :param trackletFile: name of a tracklet xml file
  :returns: list of Tracklet objects read from xml file
  """

  # convert tracklet XML data to a tree structure
  eTree = ElementTree()
  # print ('parsing tracklet file', trackletFile)
  with open(trackletFile) as f:
    eTree.parse(f)

  # now convert output to list of Tracklet objects
  trackletsElem = eTree.find('tracklets')
  tracklets = []
  trackletIdx = 0
  nTracklets = None
  for trackletElem in trackletsElem:
    #print 'track:', trackletElem.tag
    if trackletElem.tag == 'count':
      nTracklets = int(trackletElem.text)
      # print ('file contains', nTracklets, 'tracklets')
    elif trackletElem.tag == 'item_version':
      pass
    elif trackletElem.tag == 'item':
      #print 'tracklet {0} of {1}'.format(trackletIdx, nTracklets)
      # a tracklet
      newTrack = Tracklet()
      isFinished = False
      hasAmt = False
      frameIdx = None
      for info in trackletElem:
        #print 'trackInfo:', info.tag
        if isFinished:
          raise ValueError('more info on element after finished!')
        if info.tag == 'objectType':
          newTrack.objectType = info.text
        elif info.tag == 'h':
          newTrack.size[0] = float(info.text)
        elif info.tag == 'w':
          newTrack.size[1] = float(info.text)
        elif info.tag == 'l':
          newTrack.size[2] = float(info.text)
        elif info.tag == 'first_frame':
          newTrack.firstFrame = int(info.text)
        elif info.tag == 'poses':
          # this info is the possibly long list of poses
          for pose in info:
            #print 'trackInfoPose:', pose.tag
            if pose.tag == 'count':   # this should come before the others
              if newTrack.nFrames is not None:
                raise ValueError('there are several pose lists for a single track!')
              elif frameIdx is not None:
                raise ValueError('?!')
              newTrack.nFrames = int(pose.text)
              newTrack.trans  = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              newTrack.rots   = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              newTrack.states = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
              newTrack.occs   = np.nan * np.ones((newTrack.nFrames, 2), dtype='uint8')
              newTrack.truncs = np.nan * np.ones(newTrack.nFrames, dtype='uint8')
              newTrack.amtOccs = np.nan * np.ones((newTrack.nFrames, 2), dtype=float)
              newTrack.amtBorders = np.nan * np.ones((newTrack.nFrames, 3), dtype=float)
              frameIdx = 0
            elif pose.tag == 'item_version':
              pass
            elif pose.tag == 'item':
              # pose in one frame
              if frameIdx is None:
                raise ValueError('pose item came before number of poses!')
              for poseInfo in pose:
                #print 'trackInfoPoseInfo:', poseInfo.tag
                if poseInfo.tag == 'tx':
                  newTrack.trans[frameIdx, 0] = float(poseInfo.text)
                elif poseInfo.tag == 'ty':
                  newTrack.trans[frameIdx, 1] = float(poseInfo.text)
                elif poseInfo.tag == 'tz':
                  newTrack.trans[frameIdx, 2] = float(poseInfo.text)
                elif poseInfo.tag == 'rx':
                  newTrack.rots[frameIdx, 0] = float(poseInfo.text)
                elif poseInfo.tag == 'ry':
                  newTrack.rots[frameIdx, 1] = float(poseInfo.text)
                elif poseInfo.tag == 'rz':
                  newTrack.rots[frameIdx, 2] = float(poseInfo.text)
                elif poseInfo.tag == 'state':
                  newTrack.states[frameIdx] = stateFromText[poseInfo.text]
                elif poseInfo.tag == 'occlusion':
                  newTrack.occs[frameIdx, 0] = occFromText[poseInfo.text]
                elif poseInfo.tag == 'occlusion_kf':
                  newTrack.occs[frameIdx, 1] = occFromText[poseInfo.text]
                elif poseInfo.tag == 'truncation':
                    if cfg.DATA_SETS_TYPE == 'kitti':
                      newTrack.truncs[frameIdx] = truncFromText[poseInfo.text]
                    elif cfg.DATA_SETS_TYPE == 'didi2':
                      # todo :'truncation filter disable')
                      pass
                    elif cfg.DATA_SETS_TYPE == 'didi':
                      # todo :'truncation filter disable')
                      pass
                    elif cfg.DATA_SETS_TYPE == 'test':
                      pass
                    else:
                      raise ValueError('unexpected type in cfg.DATA_SETS_TYPE :{}!'.format(cfg.DATA_SETS_TYPE))
                elif poseInfo.tag == 'amt_occlusion':
                  newTrack.amtOccs[frameIdx,0] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_occlusion_kf':
                  newTrack.amtOccs[frameIdx,1] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_l':
                  newTrack.amtBorders[frameIdx,0] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_r':
                  newTrack.amtBorders[frameIdx,1] = float(poseInfo.text)
                  hasAmt = True
                elif poseInfo.tag == 'amt_border_kf':
                  newTrack.amtBorders[frameIdx,2] = float(poseInfo.text)
                  hasAmt = True
                else:
                  raise ValueError('unexpected tag in poses item: {0}!'.format(poseInfo.tag))
              frameIdx += 1
            else:
              raise ValueError('unexpected pose info: {0}!'.format(pose.tag))
        elif info.tag == 'finished':
          isFinished = True
        else:
          raise ValueError('unexpected tag in tracklets: {0}!'.format(info.tag))
      #end: for all fields in current tracklet

      # some final consistency checks on new tracklet
      if not isFinished:
        warn('tracklet {0} was not finished!'.format(trackletIdx))
      if newTrack.nFrames is None:
        warn('tracklet {0} contains no information!'.format(trackletIdx))
      elif frameIdx != newTrack.nFrames:
        warn('tracklet {0} is supposed to have {1} frames, but perser found {1}!'.format(\
            trackletIdx, newTrack.nFrames, frameIdx))
      if np.abs(newTrack.rots[:,:2]).sum() > 1e-16:
        warn('track contains rotation other than yaw!')

      # if amtOccs / amtBorders are not set, set them to None
      if not hasAmt:
        newTrack.amtOccs = None
        newTrack.amtBorders = None

      # add new tracklet to list
      tracklets.append(newTrack)
      trackletIdx += 1

    else:
      raise ValueError('unexpected tracklet info')
  #end: for tracklet list items

  # print ('loaded', trackletIdx, 'tracklets')

  # final consistency check
  if trackletIdx != nTracklets:
    warn('according to xml information the file has {0} tracklets, but parser found {1}!'.format(nTracklets, trackletIdx))

  return tracklets
#end: function parseXML

#----------------------------------------------------------------------------------
def example(kittiDir=None, drive=None):

  from os.path import join, expanduser
  import readline    # makes raw_input behave more fancy
  # from xmlParser import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

  DEFAULT_DRIVE = '2011_09_26_drive_0001'
  twoPi = 2.*np.pi

  # get dir names
  if kittiDir is None:
    kittiDir = expanduser(raw_input('please enter kitti base dir (e.g. ~/path/to/kitti): ').strip())
  if drive is None:
    drive    = raw_input('please enter drive name (default {0}): '.format(DEFAULT_DRIVE)).strip()
    if len(drive) == 0:
      drive = DEFAULT_DRIVE

  # read tracklets from file
  myTrackletFile = join(kittiDir, drive, 'tracklet_labels.xml')
  tracklets = parseXML(myTrackletFile)

  # loop over tracklets
  for iTracklet, tracklet in enumerate(tracklets):
    print ('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

    # this part is inspired by kitti object development kit matlab code: computeBox3D
    h,w,l = tracklet.size
    trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
        [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
        [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
        [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

    # loop over all data in tracklet
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
        in tracklet:

      # determine if object is in the image; otherwise continue
      if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
        continue

      # re-create 3D bounding box in velodyne coordinate system
      yaw = rotation[2]   # other rotations are 0 in all xml files I checked
      assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
      rotMat = np.array([\
          [np.cos(yaw), -np.sin(yaw), 0.0], \
          [np.sin(yaw),  np.cos(yaw), 0.0], \
          [        0.0,          0.0, 1.0]])
      cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

      # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
      #   car-centered yaw (i.e. 0 degree = same orientation as car).
      #   makes quite a difference for objects in periphery!
      # Result is in [0, 2pi]
      x, y, z = translation
      yawVisual = ( yaw - np.arctan2(y, x) ) % twoPi

    #end: for all frames in track
  #end: for all tracks
#end: function example

# when somebody runs this file as a script:
#   run example if no arg or only 'example' was given as arg
#   otherwise run parseXML
if __name__ == "__main__":
  # cmdLineArgs[0] is 'parseTrackletXML.py'
  if len(cmdLineArgs) < 2:
    example()
  elif (len(cmdLineArgs) == 2) and (cmdLineArgs[1] == 'example'):
    example()
  else:
    parseXML(*cmdLineArgs[1:])

# (created using vim - the world's best text editor)