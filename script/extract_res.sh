#!/bin/bash

file="log/250109-162348-W10000-S500-K4-varyR-rbf.log"
# log/250107-110843-varyW-S500-K4-R0.035-rbf.log
# log/250107-110843-varyW-S500-K6-R1.3-tao.log
# log/250107-110843-varyW-S5000-K2-R0.07-stk.log
# log/250107-110843-varyW-S10000-K6-R0.01-geolife.log
# log/250107-110843-W200000-S10000-K6-varyR-geolife.log
# log/250107-110843-W200000-varyS-K6-R0.01-geolife.log

grep "\[Time\] total:" "$file" | awk -F'total: ' '{print $2}' | awk '{print $1}'