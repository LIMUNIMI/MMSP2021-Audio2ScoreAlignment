#! /bin/bash

this_script_path=$(realpath $0)
this_script_dir=$(dirname $this_script_path)
ProgramFolder="$this_script_dir/Programs"

mkdir $ProgramFolder

g++ -O2 $this_script_dir/ErrorDetection_v190702.cpp -o $ProgramFolder/ErrorDetection
g++ -O2 $this_script_dir/RealignmentMOHMM_v170427.cpp -o $ProgramFolder/RealignmentMOHMM
g++ -O2 $this_script_dir/ScorePerfmMatcher_v170101_2.cpp -o $ProgramFolder/ScorePerfmMatcher
g++ -O2 $this_script_dir/midi2pianoroll_v170504.cpp -o $ProgramFolder/midi2pianoroll
g++ -O2 $this_script_dir/MusicXMLToFmt3x_v170104.cpp -o $ProgramFolder/MusicXMLToFmt3x
g++ -O2 $this_script_dir/MusicXMLToHMM_v170104.cpp -o $ProgramFolder/MusicXMLToHMM
g++ -O2 $this_script_dir/SprToFmt3x_v170225.cpp -o $ProgramFolder/SprToFmt3x
g++ -O2 $this_script_dir/Fmt3xToHmm_v170225.cpp -o $ProgramFolder/Fmt3xToHmm
g++ -O2 $this_script_dir/MatchToCorresp_v170918.cpp -o $ProgramFolder/MatchToCorresp

