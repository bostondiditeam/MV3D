#!/bin/bash

# defaults
INPUT_DIR="/data/"
OUTPUT_DIR="/data/output"
IMAGE_TAG="udacity-sdc2017-reader"
RUN_SCRIPT="/bin/bash"
INTERACTIVE="-it"

usage() { echo "Usage: $0 [-i input_dir] [-o output_dir] [-t image_tag]" 1>&2; exit 1; }
while getopts ":i:o:t:r:h" opt; do
  case $opt in
    i) INPUT_DIR=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    t) IMAGE_TAG=$OPTARG ;;
    r) RUN_SCRIPT=$OPTARG; INTERACTIVE='' ;;
    h) usage ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
shift "$((OPTIND - 1))"

echo "Running '$RUN_SCRIPT' with input dir '$INPUT_DIR', output dir '$OUTPUT_DIR', docker image '$IMAGE_TAG'..."

docker run --rm $INTERACTIVE\
  --volume="/$(pwd)/python:/python"\
  --volume="$INPUT_DIR:/data"\
  --volume="$OUTPUT_DIR:/output"\
  $IMAGE_TAG $RUN_SCRIPT "$@"
