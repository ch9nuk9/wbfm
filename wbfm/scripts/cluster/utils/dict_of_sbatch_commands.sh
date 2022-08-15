
declare -A CMD_DICT
CMD_DICT["0b"]="0b-preprocess_working_copy_of_data.py"
CMD_DICT["1"]="1-segment_video.py"
CMD_DICT["2a"]="2a-build_frame_objects.py"
CMD_DICT["2b"]="2b-match_adjacent_volumes.py"
CMD_DICT["2c"]="2c-postprocess_matches_to_tracklets.py"
CMD_DICT["3a"]="3a-track_using_superglue.py"
CMD_DICT["3b"]="3b-match_tracklets_and_tracks_using_neuron_initialization.py"
CMD_DICT["4"]="4-make_final_traces.py"
CMD_DICT["4-alt"]="alternate/4c-extract_full_traces.py"
