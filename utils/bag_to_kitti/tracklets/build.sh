#!/bin/bash
IMAGE_TAG="udacity-sdc2017-reader"

while getopts ":t:" opt; do
  case $opt in
    t) IMAGE_TAG=$OPTARG ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
shift $(expr $OPTIND - 1)

docker build -t $IMAGE_TAG .
