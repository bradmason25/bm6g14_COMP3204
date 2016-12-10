package Run3;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.analyser.ImageAnalyser;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class Run3 {
	VFSListDataset<FImage> testData;
	VFSGroupDataset<FImage> trainingImages;
	PrintWriter out;
	PrintWriter log;
	LiblinearAnnotator<FImage, String> ann;
	ClasificationVoteAgregator phow;
	long startTime;
	String TRAINING_DATA_DIR = "/home/brad/OpenIMAJ_Coursework3/training";
	String TESTING_DATA_DIR = "/home/brad/OpenIMAJ_Coursework3/testing";
	
	Run3() {
		startTime = System.currentTimeMillis();
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Starting");
		try {
			//##### If you intend to use this code make sure you set the directories below correctly
			
			System.out.println((System.currentTimeMillis()-startTime)+"ms - Gathering Files");			//Collect the training and testing images from the directories
			//trainingImages = new VFSGroupDataset<FImage>("C:/Users/brad/OpenIMAJ_CW3/training", ImageUtilities.FIMAGE_READER);
			trainingImages = new VFSGroupDataset<FImage>(TRAINING_DATA_DIR, ImageUtilities.FIMAGE_READER);
			//testData = new VFSListDataset<FImage>("C:/Users/brad/OpenIMAJ_CW3/testing", ImageUtilities.FIMAGE_READER);
			testData = new VFSListDataset<FImage>(TESTING_DATA_DIR, ImageUtilities.FIMAGE_READER);
			out = new PrintWriter("run3.txt");															//Also starts the print writer for the result file
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		Run3 r = new Run3();
		r.classifyImages();									//Start an instance of this class running
	}

	public void classifyImages() {
		
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Creating the classifier");
		phow = new ClasificationVoteAgregator(trainingImages, startTime);		//Create a new image feature extractor
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Evaluating Classifier");
		float accuracy = evaluate(trainingImages);
		System.out.println("Boasting a accuracy of: "+accuracy);
		try {
			log = new PrintWriter("log.txt");
			log.println(accuracy);
			log.close();
		} catch (FileNotFoundException e) {}
		
		
		FileObject[] files = testData.getFileObjects();		//This is needed to output the filenames to the file
		String result;
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Processing Images");
		//for(int i=0; i<testData.size(); i++) {				//Loops through the images in the testing set
		for(int i=0; i<0; i++) {
			result = files[i].getName().getBaseName()+" "+collateVotes(testData.get(i));		//Runs the voting algorithm below and stores the result to a file
			out.println(result);
			System.out.println((System.currentTimeMillis()-startTime)+" - "+result);			//Output to the terminal for debugging
		}
		out.close();										//Don't forget to close the print writer to flush the output
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Finished");
	}
	

	public float evaluate(VFSGroupDataset<FImage> images) {
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 15, 0, 15);
		int total = 0;
		int correct = 0;
		for(String key: splits.getTestDataset().keySet()) {
			for(FImage f: splits.getTestDataset().getInstances(key)) {
				total++;
				if(collateVotes(f).equals(key)) {
					correct++;
				}
			}
		}
		System.out.println(correct+" out of "+total);
		return (float) correct/total;
	}
	
	private String collateVotes(FImage f) {
		HashMap<String, Integer> votes = phow.getVotes(f);						//Hash map of the classes and the number of votes it has
		
		int max = 0;
		String maxS = "";
		for(String c: votes.keySet()) {											//Loop through the hash map to find the most voted
			if(votes.get(c)>max) {												//If there is a draw this just uses the first one
				maxS = c;
				max = votes.get(c);
			}
		}
		
		return maxS;															//Returns the most voted for
	}
	
	
	
}
