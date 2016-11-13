package uk.ac.soton.ecs.bm6g14;

import java.io.File;
import java.text.SimpleDateFormat;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.Gaussian2D;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;


public class CW2 {
	
	static MyConvolution conv;
    //private int sigma;
    //private int size;
	
	public static void main( String[] args ) {
		
		SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
    	
    	try {
    		//Retrieve the two images
    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Importing Files ");
    		//IMPORTANT - If you wish to use this you will need to change the file path to something useful for you
    		MBFImage imageA = ImageUtilities.readMBF(new File("superman.jpg"));
    		MBFImage imageB = ImageUtilities.readMBF(new File("clark.jpg"));
    		//Clone the second (high pass) for manipulation
    		MBFImage clone = imageB.clone();
    		//To make later calculations look cleaner
    		int h = imageA.getHeight();
    		int w = imageA.getWidth();
    		
    		
    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Calculating Kernel Size ");
    		//For the kernel I define sigma and thus calculate the size
    		int sigma = 10;
    		int size = (int) (8.0f * sigma + 1.0f);
    		//The size must be odd
    		if (size % 2 == 0) { size ++; }
    		
    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Creating Kernel");
    		//Produce the kernel
    		FImage kernel = Gaussian2D.createKernelImage(size, sigma);
    		
    		//Create an instance of the convolution class
    		conv = new MyConvolution(kernel.pixels);
    		
    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Convoluting first image");
    		//Convolute the low pass image
    		MBFImage lowpass = imageA.process(conv);

    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Convoluting second image");
    		//Subtract a convolution of the second image from it's unaltered original to produce the high pass
    		clone = clone.process(conv);

    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Subtracting convoluted second image from the orignal");
    		MBFImage highpass = imageB.subtract(clone);
    		//Add the pixels of the low pass and high pass together

    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Adding the highpass and lowpass together");
    		MBFImage result = lowpass.add(highpass);
    		

    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Setting edge pixels to black");
    		//Set the edge pixels to black
    		if(h==w) { //If the image is square we can reduce the time it takes to do this
    			for(int n=0; n<h-1; n++) {
    				result.setPixel(0, n, RGBColour.BLACK);
    				result.setPixel(n, 0, RGBColour.BLACK);
    			}
    		}
    		else {
    			for(int x=0; x<w-1; x++) {
    				for(int y=0; y<h-1; y++) {
    					if(y==0 || y==(h-1) || x==0 || x==(w-1)) {
    						result.setPixel(x, y, RGBColour.BLACK);
    					}
    				}
    			}
    		}
    		
    		
    		//A series of drawing functions that produce a nice display of what was produced
    		

    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Drawing images to frame");
    		MBFImage frame = new MBFImage(w*4,h*2);
    		frame.fill(RGBColour.WHITE);
    		frame.drawImage(imageA, 0, 0);
    		frame.drawText("Image A", 5, 20, HersheyFont.FUTURA_MEDIUM, 20, RGBColour.BLACK);
    		frame.drawImage(imageB, w, 0);
    		frame.drawText("Image B", w+5, 20, HersheyFont.FUTURA_MEDIUM, 20, RGBColour.BLACK);
    		frame.drawImage(lowpass, 0, h);
    		frame.drawText("A Convoluted", 5, h+20, HersheyFont.FUTURA_MEDIUM, 20, RGBColour.BLACK);
    		frame.drawImage(clone, w, h);
    		frame.drawText("B Convoluted", w+5, h+20, HersheyFont.FUTURA_MEDIUM, 20, RGBColour.BLACK);
    		frame.drawImage(highpass, w*2, 0);
    		frame.drawText("B High-pass", (w*2)+10, 20, HersheyFont.FUTURA_MEDIUM, 20, RGBColour.WHITE);
    		frame.drawImage(result, w*2, h);
    		frame.drawText("Final Result", (w*2)+10, h+20, HersheyFont.FUTURA_MEDIUM, 20, RGBColour.BLACK);
    		frame.drawImage(ResizeProcessor.halfSize(result), w*3, h);
    		frame.drawImage(ResizeProcessor.halfSize(ResizeProcessor.halfSize(result)), w*3, (int) (h*1.5));
    		frame.drawImage(ResizeProcessor.halfSize(ResizeProcessor.halfSize(ResizeProcessor.halfSize(result))), w*3, (int) (h*1.75));
    		

    		System.out.println(sdf.format(System.currentTimeMillis()) + " # Displaying frame");
    		//Finally, display the frame
    		DisplayUtilities.displayName(frame, "Result");
    		
    	} catch(Exception e){ e.printStackTrace(); }
    	

		System.out.println(sdf.format(System.currentTimeMillis()) + " # Finished");
    	
    	
    }
}
