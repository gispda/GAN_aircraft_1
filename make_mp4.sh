#!/bin/bash

avconv -f image2 -i ac/frame_%08d.png -r 30 -crf 5 -pix_fmt yuv420p -vcodec libx264 aircraft.mp4

vlc aircraft.mp4

echo "done!"
