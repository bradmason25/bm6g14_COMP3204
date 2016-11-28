package uk.ac.soton.ecs.bm6g14.ch4;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;

public class Tutorial4_1_2 {
	public static void main( String[] args ) {
		try {
			URL[] imageURLs = new URL[] {
			   new URL( "http://users.ecs.soton.ac.uk/dpd/projects/openimaj/tutorial/hist1.jpg" ),
			   new URL( "http://users.ecs.soton.ac.uk/dpd/projects/openimaj/tutorial/hist2.jpg" ), 
			   new  URL( "http://users.ecs.soton.ac.uk/dpd/projects/openimaj/tutorial/hist3.jpg" ) 
			};
			
			List<MultidimensionalHistogram> histograms = new ArrayList<MultidimensionalHistogram>();
			HistogramModel model = new HistogramModel(4, 4, 4);
			
			for( URL u : imageURLs ) {
			    model.estimateModel(ImageUtilities.readMBF(u));
			    histograms.add( model.histogram.clone() );
			}
			
			
			int hS = histograms.size();
			double distances[] = new double[hS*hS];
			int counter = 0;
			double lowest = 1.0;
			URL differentA = null;
			URL differentB = null;
			
			for(int i=0; i<hS; i++) {
				for(int j=i; j<hS; j++) {
					distances[counter] = histograms.get(i).compare(histograms.get(j), DoubleFVComparison.INTERSECTION);
					
					if(distances[counter]>0.0 && distances[counter]<lowest) {
						differentA = imageURLs[i];
						differentB = imageURLs[j];
						lowest = distances[counter];
					}
					
					counter++;
				}
			}
			DisplayUtilities.display(ImageUtilities.readMBF(differentA));
			DisplayUtilities.display(ImageUtilities.readMBF(differentB));
			//This gives a result that is the opposite to what I expected. These images to the human eye are the least similar.
			//They do not appear to have similar colours nor any similar shapes and features.
			
		} catch(Exception e) {}
	}
}
