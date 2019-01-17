#!/bin/bash
nvidia-docker run -v ${PWD}/results:/usr/src/app/results -it --rm bsch_isy bash