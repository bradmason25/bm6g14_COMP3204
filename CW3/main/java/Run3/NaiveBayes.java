package Run3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator.Mode;

public class NaiveBayes extends Classifier{
	/* Implementation of the classifier using the naive Bayes annotator
	 * 
	 */
	
	NaiveBayesAnnotator<FImage, String> annotator;
	NaiveBayes(FeatureExtractor<? extends FeatureVector, FImage> extractor) {
		annotator = new NaiveBayesAnnotator<FImage, String>(extractor, Mode.ALL);				//Initialise the naive bayes annotator
		name = "Naive Bayes";
	}
	
	public void train(GroupedDataset<String,ListDataset<FImage>, FImage> dataset) {
		annotator.train(dataset);																//Train the classifier
	}
}
