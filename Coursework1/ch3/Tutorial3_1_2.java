package uk.ac.soton.ecs.bm6g14.ch3;

import java.net.URL;
import java.util.Arrays;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;

public class Tutorial3_1_2 {
	public static void main( String[] args ) {
    	try {
			MBFImage input = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
			MBFImage copy = input.clone();

			
			FelzenszwalbHuttenlocherSegmenter<MBFImage> f = new FelzenszwalbHuttenlocherSegmenter<MBFImage>();
			List<ConnectedComponent> components = f.segment(copy);
			
			SegmentationUtilities.renderSegments(copy, components);
			DisplayUtilities.display(copy);
			
			//This method of segmentation is much slower but automatically determines the number of segments and draws them to the image.
			//IT has more accurately defined objects than the previous method so the accuracy counters the speed.
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}
