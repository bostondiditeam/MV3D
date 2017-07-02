""" ROS Bag file reading helpers
"""

from __future__ import print_function
from six import iteritems
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import fnmatch
import subprocess
import cv2
import imghdr
import yaml
import rosbag
import datetime
import pandas as pd
import numpy as np


SEC_PER_NANOSEC = 1e9
MIN_PER_NANOSEC = 6e10


def get_outdir(base_dir, name=''):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def check_image_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


class ImageBridge:

    def __init__(self):
        self.bridge = CvBridge()

    def write_image(self, outdir, msg, fmt='png'):
        results = {}
        image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
        try:
            if hasattr(msg, 'format') and 'compressed' in msg.format:
                buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
                cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
                if cv_image.shape[2] != 3:
                    print("Invalid image %s" % image_filename)
                    return results
                results['height'] = cv_image.shape[0]
                results['width'] = cv_image.shape[1]
                # Avoid re-encoding if we don't have to
                if check_image_format(msg.data) == fmt:
                    buf.tofile(image_filename)
                else:
                    cv2.imwrite(image_filename, cv_image)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imwrite(image_filename, cv_image)
        except CvBridgeError as e:
            print(e)
        results['filename'] = image_filename
        return results


def get_bag_info(bag_file, nanosec=True):
    info = yaml.load(subprocess.Popen(
        ['rosbag', 'info', '--yaml', bag_file],
        stdout=subprocess.PIPE).communicate()[0])
    if nanosec:
        if 'start' in info:
            info['start'] = int(info['start']*1e9)
        if 'end' in info:
            info['end'] = int(info['end']*1e9)
        if 'duration' in info:
            info['duration'] = int(info['duration']*1e9)
    return info


def get_topic_names(bag_info_yaml):
    topic_names = []
    topics = bag_info_yaml['topics']
    for t in topics:
        topic_names.append(t['topic'])
    return topic_names


def ns_to_str(timestamp_ns):
    secs = timestamp_ns / 1e9
    dt = datetime.datetime.fromtimestamp(secs)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')


class BagReader(object):
    def __init__(self, bagfiles, topics):
        self.bagfiles = bagfiles
        self.topics = topics

    def read_messages(self):
        for f in self.bagfiles:
            with rosbag.Bag(f, "r") as bag:
                for topic, msg, ts in bag.read_messages(topics=self.topics):
                    yield topic, msg, ts


JOIN_THRESH_NS = 10 * MIN_PER_NANOSEC


class BagSet(object):

    def __init__(self, name, bagfiles, filter_topics, metadata_path='', prefix_path=''):
        self.name = name
        self.files = sorted(bagfiles)
        self.infos = []
        self.topic_map = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self.metadata = []  # metadata will be in a list of records, each record (row) a dict of col->values
        if metadata_path:
            if os.path.exists(metadata_path):
                self._load_metadata(metadata_path)
            else:
                print('Warning: Metadata filename %s specified but not found' % metadata_path)
        self.prefix_path = prefix_path
        self._process_infos(filter_topics)

    def _load_metadata(self, metadata_path):
        metadata_df = pd.read_csv(metadata_path, header=0, index_col=None, quotechar="'")
        self.metadata = metadata_df.to_dict(orient='records')

    def _process_infos(self, filter_topics):
        for f in self.files:
            print("Extracting bag info %s" % f)
            sys.stdout.flush()
            info = get_bag_info(f)
            if 'start' not in info or 'end' not in info:
                print('Ignoring info %s without start/end time' % info['path'])
                continue
            info_start = info['start']
            info_end = info['end']
            if not self.start_time or not self.end_time:
                self._extend_range(info_start, info_end)
            elif (info_start - JOIN_THRESH_NS) <= self.end_time and self.start_time <= (info_end + JOIN_THRESH_NS):
                self._extend_range(info_start, info_end)
            else:
                print('Orphaned bag info time range, are there multiple datasets in same folder?')
                continue
            self.infos.append(info)
            filtered = [x['topic'] for x in info['topics'] if not filter_topics or x['topic'] in filter_topics]
            for x in filtered:
                self.topic_map[x].append((info['start'], info['path']))
                self.topic_map[x] = sorted(self.topic_map[x])

    def _extend_range(self, start_time, end_time):
        if not self.start_time or start_time < self.start_time:
            self.start_time = start_time
        if not self.end_time or end_time > self.end_time:
            self.end_time = end_time

    def write_infos(self, dest):
        for info in self.infos:
            info_path = os.path.splitext(os.path.basename(info['path']))[0]
            write_file = os.path.join(dest, info_path + '.yaml')
            with open(write_file, 'w') as f:
                yaml.dump(info, f)

    def get_message_count(self, topic_filter=[]):
        count = 0
        for info in self.infos:
            for topic in info['topics']:
                if not topic_filter or topic['topic'] in topic_filter:
                    count += topic['messages']
        return count

    def get_readers(self):
        readers = []
        for topic, timestamp_files in iteritems(self.topic_map):
            starts, files = zip(*timestamp_files)
            merged = False
            for r in readers:
                if r.bagfiles == files:
                    r.topics.append(topic)
                    merged = True
            if not merged:
                readers.append(BagReader(bagfiles=files, topics=[topic]))
        return readers

    def __repr__(self):
        return "start: %s, end: %s, topic_map: %s" % (self.start_time, self.end_time, str(self.topic_map))

    def get_name(self, unique=False):
        if unique and self.prefix_path:
            return '-'.join([self.prefix_path.replace('/', '_'), self.name])
        return self.name


def find_bagsets(
        directory,
        filter_topics=[],
        pattern="*.bag",
        set_per_file=False,
        metadata_filename=''):
    sets = []
    metadata_path = ''
    for root, dirs, files in os.walk(directory):
        matched_files = []
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                matched_files.append(filename)
            if metadata_filename and basename == metadata_filename:
                metadata_path = os.path.join(root, metadata_filename)
        if matched_files:
            if set_per_file:
                for f in matched_files:
                    froot, fbase = os.path.split(f)
                    relpath = os.path.relpath(froot, directory)
                    if relpath == '.' or relpath == '..':
                        relpath = ''
                    set_name = os.path.splitext(fbase)[0]
                    bag_set = BagSet(
                        set_name, [f], filter_topics, metadata_path=metadata_path, prefix_path=relpath)
                    sets.append(bag_set)
            else:
                set_name = os.path.relpath(root, directory)
                bag_set = BagSet(set_name, matched_files, filter_topics, metadata_path=metadata_path)
                sets.append(bag_set)
    return sets


class BagCursor(object):
    def __init__(self, reader):
        self.latest_timestamp = None
        self.read_count = 0
        self.done = False
        self.vals = []
        self.reader = reader
        self._iter = reader.read_messages()

    def __bool__(self):
        return not self.done

    __nonzero__ = __bool__

    # Advance cursor by one element, store element vals list
    def advance(self, n=1):
        if self.done:
            return False
        try:
            while n > 0:
                topic, msg = next(self._iter)
                self.read_count += 1
                timestamp = msg.header.stamp.to_nsec()
                if not self.latest_timestamp or timestamp > self.latest_timestamp:
                    self.latest_timestamp = timestamp
                self.vals.append((timestamp, topic, msg))
                n -= 1
        except StopIteration:
            self.done = True
        return not self.done

    # Advance cursor by relative time duration in nanoseconds
    def advance_by(self, duration_ns):
        if not self.latest_timestamp and not self.advance():
            return False
        start_time_ns = self.latest_timestamp
        while self.advance():
            elapsed = self.latest_timestamp - start_time_ns
            if elapsed >= duration_ns:
                break
        return not self.done

    # Advance cursor until specified absolute time in nanoseconds
    def advance_until(self, end_time_ns):
        while self.advance():
            if self.latest_timestamp >= end_time_ns:
                break
        return not self.done

    def collect_vals(self, dest):
        dest.extend(self.vals)
        self.vals = []

    def clear_vals(self):
        self.vals = []

    def __repr__(self):
        return "Cursor for bags: %s, topics: %s" % (str(self.reader.bagfiles), str(self.reader.topics))


class CursorGroup(object):
    def __init__(self, readers=[], cursors=[]):
        # a group can be created from readers or existing cursors,
        if readers:
            assert not cursors
            self.cursors = [BagCursor(r) for r in readers]
        elif cursors:
            self.cursors = cursors

    def __bool__(self):
        for c in self.cursors:
            if c:
                return True
        return False

    __nonzero__ = __bool__

    def advance(self, n=1):
        all_done = True
        for c in self.cursors:
            if c and c.advance(n):
                all_done = False
        return not all_done

    # Advance all cursors by specified duration
    # Risk of cursors drifting over time from each other
    def advance_by(self, duration_ns=1*SEC_PER_NANOSEC):
        all_done = True
        for c in self.cursors:
            if c and c.advance_by(duration_ns):
                all_done = False
        return not all_done

    # Advance all cursors up to same end time
    def advance_until(self, end_time_ns):
        all_done = True
        for c in self.cursors:
            if c and c.advance_until(end_time_ns):
                all_done = False
        return not all_done

    # Advance the first ready cursor in group by specified amount and bring the reset
    # up to same resulting end time.
    # Risk of pulling in large amounts of data if leading stream has a large gap.
    def advance_by_until(self, duration_ns=1*SEC_PER_NANOSEC):
        all_done = True
        end_time_ns = None
        for c in self.cursors:
            ready = False
            if c:
                if not end_time_ns:
                    ready = c.advance_by(duration_ns)
                    end_time_ns = c.latest_timestamp
                else:
                    ready = c.advance_until(end_time_ns)
            if ready:
                all_done = False
        return not all_done

    def collect_vals(self, dest):
        for c in self.cursors:
            c.collect_vals(dest)
