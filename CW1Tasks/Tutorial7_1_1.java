package uk.ac.soton.ecs.bm1;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.math.geometry.shape.Ellipse;
import org.openimaj.video.Video;
import org.openimaj.video.VideoDisplay;
import org.openimaj.video.VideoDisplayListener;
import org.openimaj.video.xuggle.XuggleVideo;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) {
    	try {
    	
	    	Video<MBFImage> video;
	    	
	    	video = new XuggleVideo(new URL("http://static.openimaj.org/media/tutorial/keyboardcat.flv"));
	    	
	    	//VideoDisplay<MBFImage> display = VideoDisplay.createVideoDisplay(video);
	    	
	    	/*
	    	for (MBFImage mbfImage : video) {
	    		DisplayUtilities.displayName(mbfImage.process(new CannyEdgeDetector()), "videoFrames");
	    	}
	    	*/
	    	
	    	VideoDisplay<MBFImage> display = VideoDisplay.createVideoDisplay(video);
	    	display.addVideoListener(
	    			new VideoDisplayListener<MBFImage>() {
	    				public void beforeUpdate(MBFImage frame) {
	    					frame.processInplace(new CannyEdgeDetector());
	    				}
	    				
	    				public void afterUpdate(VideoDisplay<MBFImage> display) {
	    				}
	    			});
	    	
	    	VideoDisplay<MBFImage> displayB = VideoDisplay.createVideoDisplay(video);
	    	displayB.addVideoListener(
	    			new VideoDisplayListener<MBFImage>() {
	    				public void beforeUpdate(MBFImage frame) {
	    					frame.getBand(2).fill(0f);
	    					frame.getBand(0).fill(0f);
	    				}
	    				
	    				public void afterUpdate(VideoDisplay<MBFImage> displayB) {
	    				}
	    			});
    	} catch(Exception e) {}
    }
}
