package Run3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;

import de.bwaldvogel.liblinear.SolverType;

public class LibLinear extends Classifier{
	/*
	 * Classifier implementation using the liblinear annotator
	 * Using tuned values for C and epsilon
	 */
	LiblinearAnnotator<FImage, String> annotator;
	LibLinear(FeatureExtractor<? extends FeatureVector, FImage> extractor) {
		annotator = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);	//Initialise the lib linear annotator
		name = "Lib Linear";
	}
	
	public void train(GroupedDataset<String,ListDataset<FImage>, FImage> dataset) {
		annotator.train(dataset);																									//Train the classifier
	}

	@Override
	protected org.openimaj.experiment.evaluation.classification.Classifier<String, FImage> getAnnotator() {
		return annotator;
	}
}
