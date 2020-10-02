/*
 * Macro template to process multiple images in a folder
 */

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".ome.tiff") suffix
#@ int (label = "Number of files", value=10) max

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < max; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	print("Processing: " + input + File.separator + file);
	print("Saving to: " + output);
	// only proceeds if the file is there
	if(File.exists(input + File.separator + file)){
	// opens the image and gives it a temporary name
	open(input + File.separator + file);
	rename("raw_volume");
	
	selectWindow("raw_volume");
	run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");
	
	// saves the image into the sub directory as: aligned_..max
	saveAs("tiff", output + File.separator + "aligned_" + file);
	// closes all images before the next iteration
	while(nImages > 0) close();
	} else print("A file was missing: " + input);
}
