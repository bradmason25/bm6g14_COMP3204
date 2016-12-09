package Run3;

import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator.Mode;

public class NaiveBayes {
	NaiveBayesAnnotator<FImage, String> annotator;
	NaiveBayes(FeatureExtractor<? extends FeatureVector, FImage> extractor) {
		annotator = new NaiveBayesAnnotator<FImage, String>(extractor, Mode.ALL);				//Initialise the naive bayes annotator
	}
	
	public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> dataset) {
		annotator.train(dataset);																//Train the classifier
	}
	
	public String getVote(FImage f) {
		return annotator.classify(f).getPredictedClasses().iterator().next();					//Get the predicted class for an image
	}
	
	public float evaluate(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> dataset) {
		ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
				new ClassificationEvaluator<CMResult<String>, String, FImage>(annotator, dataset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));	//Produce an evaluator
		
		Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);
		
		return Float.parseFloat(result.getSummaryReport().substring(12, 17));					//Parse the resultant accuracy given by the evaluator
	}
}
