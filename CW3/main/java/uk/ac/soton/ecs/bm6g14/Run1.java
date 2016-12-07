package uk.ac.soton.ecs.bm6g14;

import java.util.ArrayList;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class Run1 {
	/*
	 * Run #1: You should develop a simple k-nearest-neighbour classifier using the “tiny image” feature. The “tiny image” feature is one of the simplest
	 * possible image representations. One simply crops each image to a square about the centre, and then resizes it to a small, fixed resolution
	 * (we recommend 16x16). The pixel values can be packed into a vector by concatenating each image row. It tends to work slightly better if the tiny
	 * image is made to have zero mean and unit length. You can choose the optimal k-value for the classifier.
	 */
	
	private float[] tinyImage(FImage image) {
		/*
		 * This method takes the FImage as input,
		 * Crops is square about the centre
		 * Resizes it to a 16x16 resolution image
		 * Concatenates each row into a single vector
		 * 
		 * ***** Make it have zero mean and unit length *****
		 * 
		 * Then returns it
		 */
		int vectorSize = 16*16;
		float[] vector;
		
		//Crop the image square
		image = image.extractCenter(256, 256);
		//Reduce the image to a 16th size to get 16x16
		image = ResizeProcessor.halfSize(ResizeProcessor.halfSize(ResizeProcessor.halfSize(ResizeProcessor.halfSize(image))));
		
		//Find the mean
		float mean = image.sum()/vectorSize;
		//Find the standard deviation
		float var = 0;
		for(int y=0;y<16;y++) {
			for(float f: image.pixels[y]) {
				var+=(f-mean)*(f-mean);
			}
		}
		float sd = var/vectorSize;
		
		image.subtractInplace(mean);
		image.divideInplace(sd);
		
		vector = image.getFloatPixelVector();
		
		return vector;
	}

	public String knn(int k, FImage testImage) throws FileSystemException {
		//Loop through training data
		//Pick out the k nearest matches
		//The majority vote of class is the class of the image
		float[] t = tinyImage(testImage);

		String[] nn = new String[k];
		int[] distances = new int[k];
		int ni = 0;
		int maxDistance = Integer.MIN_VALUE;
		
		VFSGroupDataset<FImage> trainingImages = new VFSGroupDataset<FImage>("zip:file:/home/brad/OpenIMAJ_Coursework3/training.zip", ImageUtilities.FIMAGE_READER);
		
		GroupedDataset<String, VFSListDataset<FImage>, FImage> data;
		
		for (final Entry<String, VFSListDataset<FImage>> entry: trainingImages.entrySet()) {
			for(FImage image: entry.getValue()) {
				float[] v = tinyImage(image);
				int distance = 0;
				for(int i=0;i<v.length;i++) {
					distance += Math.pow(Math.abs(t[i]-v[i]),2);
				}
				if(ni<k) {
					nn[ni] = entry.getKey();
					distances[ni] = distance;
					ni++;
					maxDistance = Math.max(maxDistance, distance);
				}
				else if(distance<maxDistance) {
					int i = findHighestIndex(distances,distance);
					maxDistance = updateMax(distances);
					distances[i] = distance;
					nn[i] = entry.getKey();
				}
				
				
			}
		}
		
		//Vote for the class
		ArrayList<String> uniqueS = new ArrayList<String>();
		ArrayList<Integer> uniqueI = new ArrayList<Integer>();
		for(int i=0;i<k;i++) {
			if(uniqueS.contains(nn[i])) {
				int index = uniqueS.indexOf(nn[i]);
				uniqueI.set(index, uniqueI.get(index)+1);
			}
			else {
				uniqueS.add(nn[i]);
				uniqueI.add(distances[i]);
			}
		}
		int max = 0;
		int maxi = 0;
		for(int i=0;i<uniqueS.size();i++) {
			if(uniqueI.get(i)>max) {
				max = uniqueI.get(i);
				maxi = i;
			}
		}
		
		return uniqueS.get(maxi);
	}

	private int findHighestIndex(int[] list, int item) {
		int highest = 0;
		int index = 0;
		for(int n=0;n<list.length;n++) {
			if(list[n]>highest) {
				highest = list[n];
				index = n;
			}
		}
		return index;
	}
	
	private int updateMax(int[] list) {
		int max = 0;
		for(int n=0;n<list.length;n++) {
			if(list[n]>max) {
				max = list[n];
			}
		}
		return max;
	}

}
