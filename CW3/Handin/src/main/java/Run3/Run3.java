package Run3;

import java.io.PrintWriter;
import java.util.HashMap;

import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

public class Run3 {
	VFSListDataset<FImage> testData;
	VFSGroupDataset<FImage> trainingImages;
	PrintWriter out;
	Log log;
	LiblinearAnnotator<FImage, String> ann;
	ClassificationVoteAggregator phow;
	GroupedRandomSplitter<String, FImage> splits;
	String TRAINING_DATA_DIR = "/home/brad/OpenIMAJ_Coursework3/training_aug";
	String TESTING_DATA_DIR = "/home/brad/OpenIMAJ_Coursework3/testing";
	
	public Run3() {
		try {
			//##### If you intend to use this code make sure you set the directories correctly
			log = new Log("run3_"+System.currentTimeMillis()+".txt");
			log.log("Starting");	
			trainingImages = new VFSGroupDataset<FImage>(TRAINING_DATA_DIR, ImageUtilities.FIMAGE_READER); //Collect the training and testing images from the directories
			testData = new VFSListDataset<FImage>(TESTING_DATA_DIR, ImageUtilities.FIMAGE_READER);
			
			splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 15, 0, 15);
			
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
		log.log("Creating Classifier");
		phow = new ClassificationVoteAggregator(splits.getTrainingDataset(), splits.getTestDataset(), log);		//Create a new image feature extractor
		log.log("Evaluating Classifier");
		float accuracy = evaluate(trainingImages);
		System.out.println("Boasting an accuracy of: "+accuracy);
		log.log("Overall Accuracy: "+accuracy);
		
		
		FileObject[] files = testData.getFileObjects();		//This is needed to output the filenames to the file
		String result;
		log.log("Processing Images");
		for(int i=0; i<testData.size(); i++) {				//Loops through the images in the testing set
		//for(int i=0; i<0; i++) {
			result = files[i].getName().getBaseName()+" "+collateVotes(testData.get(i));		//Runs the voting algorithm below and stores the result to a file
			out.println(result);
			log.log(result);																	//Output to the terminal for debugging
		}
		out.close();										//Don't forget to close the print writer to flush the output
		log.log("Finished");
		log.close();
	}
	

	public float evaluate(VFSGroupDataset<FImage> images) {
		
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
