package Run3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LinearSVMAnnotator;

public class SVM extends Classifier{

	/*
	 * Implementation of the classifier using the linear SVM annotator
	 */
	
	LinearSVMAnnotator<FImage, String> annotator;
	SVM(FeatureExtractor<? extends FeatureVector, FImage> extractor) {
		annotator = new LinearSVMAnnotator<FImage, String>(extractor);				//Initialise the SVM annotator
		name = "SVM";
	}
	
	public void train(GroupedDataset<String,ListDataset<FImage>, FImage> dataset) {
		annotator.train(dataset);																//Train the classifier
	}
	
	protected org.openimaj.experiment.evaluation.classification.Classifier<String, FImage> getAnnotator() {
		return annotator;
	}
}
