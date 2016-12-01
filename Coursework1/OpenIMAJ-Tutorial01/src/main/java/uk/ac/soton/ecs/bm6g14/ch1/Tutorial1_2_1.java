package uk.ac.soton.ecs.bm6g14.ch1;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

public class Tutorial1_2_1 {
    public static void main( String[] args ) {
    	//Create an image
        MBFImage image = new MBFImage(300,70, ColourSpace.RGB);

        //Fill the image with white
        image.fill(RGBColour.WHITE);
        		        
        //Render some test into the image
        image.drawText("The Cake is a Lie", 10, 60, HersheyFont.MUSIC, 50, RGBColour.BLUE);

        //Apply a Gaussian blur
        image.processInplace(new FGaussianConvolve(2f));
        
        //Display the image
        DisplayUtilities.display(image);
    }
}