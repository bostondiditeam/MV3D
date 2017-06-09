import mv3d
from net.processing.boxes3d import boxes3d_decompose


LIDAR_TOP_SHAPE = (400, 400, 8) # to be decided
LIDAR_FRONT_SHAPE = () # to be decided.
RGB_SHAPE = (1096, 1368, 3) # to be decided

# build predictor
def build_predictor(top_shape, front_shape, rgb_shape, tag=None):
    predictor = mv3d.Predictor(top_shape, front_shape, rgb_shape, tag=tag)
    return predictor


def pred_one_frame(predictor, top_shape, front_shape, rgb, timestamp):
    boxes3d, probs = predictor(top_shape, front_shape, rgb)

    if len(boxes3d) != 0:
        translation, size, rotation = boxes3d_decompose(boxes3d[:, :, :])
        return translation, size, rotation, timestamp
    else:
        return None, None, None, None


def call_mv3d_predictor_demo():
    assembler = build_assembler() # todo call assembler class or a function, according to real implementation
    preprocessor = build_preprocessor() # todo call data preprocessor
    mv3d_predictor = build_predictor(LIDAR_TOP_SHAPE, LIDAR_FRONT_SHAPE, RGB_SHAPE)

    # get the top, front, rgb and timestamp data on the fly.
    while True: # Change to call back function accordingly.
        lidar, rgb, timestamp = assembler.get_synced_data() # todo
        lidar_top, lidar_front, rgb, timestamp = preprocessor.process(lidar, rgb, timestamp) # todo
        translation, size, rotation, timestamp = pred_one_frame(mv3d_predictor, lidar_top, lidar_front, rgb, timestamp)
        publish_detectors(translation, size, rotation, timestamp) # todo publish the detections


if __name__ == '__main__':
    call_mv3d_predictor_demo()