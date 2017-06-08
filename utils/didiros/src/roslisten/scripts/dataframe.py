class OneFrameData():

    def __init__(self):
        self.cap_r = None
        self.cap_f = None
        self.obs_r = None
        self.velodyne_points = None
        self.image_raw = None
        self.complete = False

    def clearData(self):
        self.cap_r = None
        self.cap_f = None
        self.obs_r = None
        self.velodyne_points = None
        self.image_raw = None
        self.complete = False

    def addData(self, data, type):
        if type == 'cap_r':
            self.cap_r = data
        elif type == 'cap_f':
            self.cap_f = data
        elif type == 'obs_r':
            self.obs_r = data
        elif type == 'velodyne_points':
            self.velodyne_points = data
        elif type == 'image_raw':
            self.image_raw = data
        self.checkComplete()

    def checkComplete(self):
        # if ((self.cap_r is not None) and
        #     (self.cap_f is not None) and
        #     (self.obs_r is not None) and
        #     (self.velodyne_points is not None) and
        #     (self.image_raw is not None)):
        #     self.complete = True
        if ((self.velodyne_points is not None) and
            (self.image_raw is not None)):
            self.complete = True
        else:
            self.complete = False
        return self.complete
