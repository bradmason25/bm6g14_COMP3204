package uk.ac.soton.ecs.bm6g14.ch7;

import java.net.URL;

import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.video.Video;
import org.openimaj.video.VideoDisplay;
import org.openimaj.video.VideoDisplayListener;
import org.openimaj.video.xuggle.XuggleVideo;

public class Tutorial7_1_1 {
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
	    	//The edge detector is noticeably slower to process each frame
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
