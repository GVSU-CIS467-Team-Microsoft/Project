#!/bin/bash
#mkdir stage1
cd ..
cd stage2half1
for f in *; do
    if [[ -d $f ]]; then
    	cd /home/gvuser/stage2
        ln -s /home/gvuser/stage2half1/$f $f
        cd /home/gvuser/stage2half1
    fi
done

cd ..
cd stage2half2
for f in *; do
    if [[ -d $f ]]; then
    	cd /home/gvuser/stage2
        ln -s /home/gvuser/stage2half2/$f $f
        cd /home/gvuser/stage2half2
    fi
done
