package uk.ac.soton.ecs.bm6g14;

import java.text.SimpleDateFormat;

import org.openimaj.image.FImage;
import org.openimaj.image.processor.SinglebandImageProcessor;

public class MyConvolution implements SinglebandImageProcessor<Float, FImage> {
	private float[][] kernel;
	
	private int irows;
	private int icols;
	private int trows;
	private int tcols;
	private int trhalf;
	private int tchalf;
	private int modx;
	private int mody;

	public MyConvolution(float[][] kernel) {
		this.kernel = kernel;
	}

	@Override
	public void processImage(FImage image) {
		
		// convolve image with kernel and store result back in image
		
		//Define some variables about the image for later calculations
		irows = image.getRows();
		icols = image.getCols();
		trows = kernel.length;
		tcols = kernel[0].length;
		
		//Produce a temporary black image to store the convoluted pixels
		float temp[][] = new float[image.height][image.width];
		
		trhalf = (int) Math.floor(trows/2);
		tchalf = (int) Math.floor(tcols/2);
		
		//Loop through all pixels of the original image
		for (int x = 1; x<icols-1; x++) {
			for (int y = 1; y<irows-1; y++) {
				//Reset the sum
				float sum = 0;
					//Loop through all points within the kernel
					for (int iwin = 1; iwin<trows; iwin++) {
						for (int jwin = 1; jwin<tcols; jwin++) {
							try {
								//Add to the sum the value of the current image pixel multiplied by the weighting in that location of the kernel
								modx = x+iwin-trhalf-1;
								mody = y+jwin-tchalf-1;
								sum += image.pixels[mody][modx]*kernel[jwin][iwin]; 
							} catch(Exception e) {}
						}
					}
				//Set the corresponding blacked out image's pixel to the final summed value
				temp[y][x] = sum;
			}
		}
		//Normalise and return the convoluted image
		FImage ret = new FImage(temp);
		image.internalAssign(ret.normalise());
	}
}
