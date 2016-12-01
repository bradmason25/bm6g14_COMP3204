package uk.ac.soton.ecs.bm6g14.ch2;

import java.net.URL;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.math.geometry.shape.Ellipse;

public class Tutorial2_1_1 {
    public static void main( String[] args ) {
    	try {
			MBFImage image = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
			
			System.out.println(image.colourSpace);
		
			//By labelling the display "Image" it remains on the same display even when changing image
			DisplayUtilities.displayName(image, "Image");
			DisplayUtilities.displayName(image.getBand(0), "Image");
			
			MBFImage clone = image.clone();
			//Set any Green and Blue colour to black
			clone.getBand(1).fill(0f);
			clone.getBand(2).fill(0f);
			DisplayUtilities.displayName(clone, "Image");
			
			//Find edges
			image.processInplace(new CannyEdgeDetector());
			
			//Draw shapes onto the edge detected image
			image.drawShapeFilled(new Ellipse(700f, 450f, 20f, 10f, 0f), RGBColour.WHITE);
			image.drawShapeFilled(new Ellipse(650f, 425f, 25f, 12f, 0f), RGBColour.WHITE);
			image.drawShapeFilled(new Ellipse(600f, 380f, 30f, 15f, 0f), RGBColour.WHITE);
			image.drawShapeFilled(new Ellipse(500f, 300f, 100f, 70f, 0f), RGBColour.WHITE);
			image.drawText("OpenIMAJ is", 425, 300, HersheyFont.ASTROLOGY, 20, RGBColour.BLACK);
			image.drawText("Awesome", 425, 330, HersheyFont.ASTROLOGY, 20, RGBColour.BLACK);
			DisplayUtilities.displayName(image, "Image");
		} catch (Exception e) {
			e.printStackTrace();
		}
    	
    	
    }
}
