#!/bin/bash
find -maxdepth 1 -type d | while read -r dir; do printf "%s:\t" "$dir"; find "$dir" | wc -l; done
