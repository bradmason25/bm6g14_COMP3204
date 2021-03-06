package uk.ac.soton.ecs.bm6g14.ch5;

import java.net.URL;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.MatchingUtilities;
import org.openimaj.feature.local.matcher.MultipleMatchesMatcher;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.transforms.estimation.RobustAffineTransformEstimator;
import org.openimaj.math.model.fit.RANSAC;

public class Tutorial5_1_1 {
	public static void main(String[] args) {
		try {
			MBFImage query = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));
			MBFImage target = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/target.jpg"));
			
			DoGSIFTEngine engine = new DoGSIFTEngine();	
			LocalFeatureList<Keypoint> queryKeypoints = engine.findFeatures(query.flatten());
			LocalFeatureList<Keypoint> targetKeypoints = engine.findFeatures(target.flatten());
			
			//LocalFeatureMatcher<Keypoint> matcher = new BasicMatcher<Keypoint>(80);
			//LocalFeatureMatcher<Keypoint> matcher = new BasicTwoWayMatcher<Keypoint>();
			LocalFeatureMatcher<Keypoint> matcher = new MultipleMatchesMatcher<Keypoint>(3,0.01);
			matcher.setModelFeatures(queryKeypoints);
			matcher.findMatches(targetKeypoints);
			
			MBFImage basicMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
			DisplayUtilities.display(basicMatches);
			
			RobustAffineTransformEstimator modelFitter = new RobustAffineTransformEstimator(5.0, 1500, new RANSAC.PercentageInliersStoppingCondition(0.5));
			matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(new FastBasicKeypointMatcher<Keypoint>(8), modelFitter);

			matcher.setModelFeatures(queryKeypoints);
			matcher.findMatches(targetKeypoints);

			MBFImage consistentMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);

			//DisplayUtilities.display(consistentMatches);
			
			target.drawShape(query.getBounds().transform(modelFitter.getModel().getTransform().inverse()), 3,RGBColour.BLUE);
			//DisplayUtilities.display(target); 
		} catch(Exception e) {}
	}
	
	//The BasicTwoWayMatcher performed better than the BasicMatcher but still made a large amount of mistakes
	
	//The MultipleMatchesMatcher was much worse, I tried playing with the values but was unable to find values that gave an accurate result. In fact it appeared
	//to favour mistakes more than correct results.
}
