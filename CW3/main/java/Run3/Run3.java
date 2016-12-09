package Run3;

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
	static VFSListDataset<FImage> testData;
	VFSGroupDataset<FImage> trainingImages;
	PrintWriter out;
	LiblinearAnnotator<FImage, String> ann;
	PHOW phow;
	long startTime;
	
	Run3() {
		startTime = System.currentTimeMillis();
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Starting");
		try {
			//##### If you intend to use this code make sure you set the directories below correctly
			
			System.out.println((System.currentTimeMillis()-startTime)+"ms - Gathering Files");			//Collect the training and testing images from the directories
			//trainingImages = new VFSGroupDataset<FImage>("C:/Users/brad/OpenIMAJ_CW3/training", ImageUtilities.FIMAGE_READER);
			trainingImages = new VFSGroupDataset<FImage>("/home/brad/OpenIMAJ_Coursework3/training", ImageUtilities.FIMAGE_READER);
			//testData = new VFSListDataset<FImage>("C:/Users/brad/OpenIMAJ_CW3/testing", ImageUtilities.FIMAGE_READER);
			testData = new VFSListDataset<FImage>("/home/brad/OpenIMAJ_Coursework3/testing", ImageUtilities.FIMAGE_READER);
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
		 phow = new PHOW(trainingImages, startTime);		//Create a new image feature extractor
		
		
		FileObject[] files = testData.getFileObjects();		//This is needed to output the filenames to the file
		String result;
		for(int i=0; i<testData.size(); i++) {				//Loops through the images in the testing set
			result = files[i].getName().getBaseName()+" "+collateVotes(testData.get(i));		//Runs the voting algorithm below and stores the result to a file
			out.println(result);
			System.out.println(result);															//Output to the terminal for debugging
		}
		out.close();										//Don't forget to close the print writer to flush the output
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Finished");
	}
	
	
	private String collateVotes(FImage f) {
		HashMap<String, Integer> votes = new HashMap<String, Integer>();		//Hash map of the classes and the number of votes it has
		ArrayList<String> newVotes = phow.getVotes(f);							//Collects votes from the feature extractor with the different classifiers
		for(String vote: newVotes) {											//Loops through the votes and counts them in the hash map
			try {
				votes.put(vote, votes.get(vote)+1);
			} catch(Exception e) {
				votes.put(vote, 1);
			}
		}
		
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
