package uk.ac.soton.ecs.bm6g14;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;

import org.apache.commons.vfs2.FileObject;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

public class Run1 {
	/* This class uses the tiny image feature to extract features for a knn classifier from the OpenIMAJ library.
	 * There is also an evaluation method here that I used to tweak the k value to find the optimal k.
	 */
	public static void main(String[] args) {
		Run1 r = new Run1();
		r.run();
	}

	Run1() {
	}
	public void run() {
		VFSGroupDataset<FImage> trainingImages = null;
		VFSListDataset<FImage> testingImages = null;
		GroupedRandomSplitter<String, FImage> splits = null;
		try {
			trainingImages = new VFSGroupDataset<FImage>("/home/brad/OpenIMAJ_Coursework3/training_aug", ImageUtilities.FIMAGE_READER);
			testingImages = new VFSListDataset<>("/home/brad/OpenIMAJ_Coursework3/testing", ImageUtilities.FIMAGE_READER);
			splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 450, 0, 50);
		} catch (FileSystemException e) {e.printStackTrace();}
		
		FeatureExtractor<FloatFV, FImage> tinyImage = new TinyImage();
		//If uncommented the line below will evaluate different values of k to find the optimal value
		//int k = evaluateK(splits.getTestDataset(), splits.getTrainingDataset(), tinyImage);
		//From past use I have bypassed this for speed to the optimal value of 1
		int k = 1;
		
		//I use the Euclidean distance comparator for the float vectors produced by the tiny image feature extractor
		KNNAnnotator<FImage, String, FloatFV> knn = new KNNAnnotator<FImage, String, FloatFV>(tinyImage, FloatFVComparison.EUCLIDEAN, k);
		knn.train(splits.getTrainingDataset());
		
		//This array is necessary to know the name of the image files when outputting to a text file 
		FileObject[] files = testingImages.getFileObjects();
		PrintWriter out = null;
		try {
			out = new PrintWriter("run1.txt");
		} catch (Exception e) {}
		
		
		String result;
		for(int i=0; i<testingImages.size(); i++) {
			result = files[i].getName().getBaseName()+" "+knn.classify(testingImages.get(i)).getPredictedClasses().iterator().next();
			out.println(result);
		}
		
				
		out.close();
		
		
	}
	
	@SuppressWarnings("unused")
	private int evaluateK(GroupedDataset<String, ListDataset<FImage>, FImage> testing,GroupedDataset<String, ListDataset<FImage>, FImage> training, FeatureExtractor<FloatFV, FImage> tinyImage) {
		/*
		 * This is the evaluation method for finding the optimal k
		 * It loops through k values 1 to 9, 5 times each and takes the average accuracy from an evaluator
		 * It then takes the highest average accuracy and returns that as the optimal k
		 */
		
		PrintWriter log = null;
		ArrayList<Float> ks = new ArrayList<>();
		try {
			 log = new PrintWriter("knnlog.txt");
		} catch (Exception e) {}
		for(int k=1;k<10;k++) {
			float total = 0f;
			for(int i=0;i<5;i++) {
				KNNAnnotator<FImage, String, FloatFV> knn = new KNNAnnotator<FImage, String, FloatFV>(tinyImage, FloatFVComparison.EUCLIDEAN, k);
				knn.train(training);
				ClassificationEvaluator<CMResult<String>, String, FImage> eval = 
						new ClassificationEvaluator<CMResult<String>, String, FImage>(knn, testing, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
				Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
				CMResult<String> result = eval.analyse(guesses);
				//Unfortunately, I was unable to find how to extract the accuracy directly
				//So I extract it as a substring from the summary
				//Odd but it works and I can't find another solution
				total += Float.valueOf(result.getSummaryReport().substring(12, 17));
			}
			ks.add(total/5);
			System.out.println("For K = "+k+" - Average accuracy = "+(float) (total/5));
			log.println("For K = "+k+" - Average accuracy = "+(float) (total/5));
		}
		log.close();
		
		return ks.indexOf(Collections.max(ks));
	}
}
