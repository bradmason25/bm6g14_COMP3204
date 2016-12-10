package uk.ac.soton.ecs.bm6g14;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class TinyImage implements FeatureExtractor<FloatFV, FImage>{

	public FloatFV extractFeature(FImage image) {
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
		int size = Math.min(image.height, image.width);
		image = image.extractCenter(size, size);
		//Reduce the image to a 16th size to get 16x16
		image = ResizeProcessor.resample(image, 16, 16);
		
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
		
		return new FloatFV(vector);
	}

}
