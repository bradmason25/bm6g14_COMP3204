package Run3;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;


public class ClasificationVoteAgregator {
	/*
	 * This class builds the feature extractor, sets up the individual classifiers and organises their votes
	 * 
	 * In the constructor I set up the boiler plate code that uses OpenIMAJ classes to create a feature extractor
	 * Specifically the Pyramid Dense SIFT with a Homogenous kernel map
	 * It produces and trains the classifier and also evaluates them to generate a weighting
	 * 
	 */
	
	
	
	
	VFSGroupDataset<FImage> trainingImages;
	ArrayList<Classifier> annotators = new ArrayList<>();
	long startTime;
	float llweight;
	float nbweight;
	float svmweight;
	float pweight;
	PrintWriter log;

	ClasificationVoteAgregator(VFSGroupDataset<FImage> trainingImages, long startTime) {
		//This function gets the training data set and creates the annotators which are then evaluated
		try {
			log = new PrintWriter("log.txt");
			log.println("Augmented Data");
		} catch (FileNotFoundException e) {}
		
		this.startTime = startTime;
		this.trainingImages = trainingImages;
		
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 15, 0, 15);	//Splitter for evaluating the model
		
		FeatureExtractor<? extends FeatureVector, FImage> extractor = getExtractor(splits);
		
		annotators.add(new LibLinear(extractor));
		annotators.add(new NaiveBayes(extractor));
		annotators.add(new SVM(extractor));
		annotators.add(new Patches());
		
		trainAnnotators(annotators, splits.getTrainingDataset());
		
		evaluateAnnotators(annotators, splits.getTestDataset());
		
		log.close();
		
	}
	
	private void trainAnnotators(ArrayList<Classifier> classifiers, GroupedDataset<String,ListDataset<FImage>, FImage> dataset) {
		//takes the given list of classifiers and trains them
		for(Classifier c: classifiers) {
			c.train(dataset);
		}
	}
	
	private void evaluateAnnotators(ArrayList<Classifier> classifiers,GroupedDataset<String,ListDataset<FImage>, FImage> dataset) {
		//evaluate the annotator performance so that we can weight the effective vote
		for(Classifier c: classifiers) {
			float weight = c.evaluate(dataset);
			c.setWeight(weight);
			log.println(c.name+": "+weight);
		}
	}
	
	private FeatureExtractor<? extends FeatureVector, FImage> getExtractor(GroupedRandomSplitter<String, FImage> splits) {
		//generates the dense sift feature extractor using the tuned values from the image set
		//the dense features are quantised by the assigner
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Creating PyramidDenseSIFT");
		DenseSIFT dsift = new DenseSIFT(5, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);										//Created a Pyramid Dense SIFT
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training Quantiser");
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), pdsift);			//Trains Pyramid Dense SIFT with K Means Clustering
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Done");
		
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);									//Creates a new instance of PHOW extractor below
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Wrapping Extractor");
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		return hkm.createWrappedExtractor(extractor); 																		//Wraps extractor in a homogenous kernel map
	}
	
	public HashMap<String, Integer> getVotes(FImage f) {
		//store the votes for each classification label
		HashMap<String, Integer> votes = new HashMap<String, Integer>();
		for(Classifier c: annotators) {										//Here I create hash maps of the votes from each classifier
			votes.put(c.getVote(f), (int) (c.getWeight()*100));				//The number of votes depends on the accuracy of the classifier acting as weighting
		}
		return votes;
	}
	
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, PyramidDenseSIFT<FImage> pdsift)
	{
		//generate the features from the images and keep the first 10,000
		//then use a KDTree to quantise the features
	    int siftFeatures = 10000;//10000
	    List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

	    for (FImage rec : groupedDataset) {
	        FImage img = rec.getImage();

	        pdsift.analyseImage(img);
	        allkeys.add(pdsift.getByteKeypoints(0.005f));
	    }
	    
	    if (allkeys.size() > siftFeatures)
	        allkeys = allkeys.subList(0, siftFeatures);

	    ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);//300
	    DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
	    ByteCentroidsResult result = km.cluster(datasource);

	    return result.defaultHardAssigner();
	}
	
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
		//inner class for the feature extractor
		//uses tuned values for visual word aggregation
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage object) {
	        FImage image = object.getImage();
	        pdsift.analyseImage(image);

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 2);

	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}
	
	
}
