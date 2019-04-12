#!/bin/bash

mkdir -p results

for OPT in on off; do
	DIR="build_trace_$OPT"
	if [ -d "$DIR" ]; then rm -Rf $DIR; fi
	mkdir -p $DIR
	cd $DIR

	#echo Running cmake...
	cmake .. -DTRACE_STORES=$OPT 		#2>&1 >/dev/null

	#echo Running make fft.rungen...
	make -j4 fft.rungen 			#2>&1 >/dev/null

	#echo Running fft.rungen...
	./fft.rungen pattern=../results/pattern_$OPT.png filtered=../results/tracing_$OPT.png --output_extents=[300,300] #>/dev/null

	cd ..
done


