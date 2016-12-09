package Run3;

import java.util.ArrayList;
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
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
//import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator.Mode;
//import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;

public class PHOW {
	VFSGroupDataset<FImage> trainingImages;
	LibLinear llann;
	NaiveBayes nbann;
	long startTime;
	float llweight;
	float nbweight;

	PHOW(VFSGroupDataset<FImage> trainingImages, long startTime) {
		this.startTime = startTime;
		this.trainingImages = trainingImages;
		
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 15, 0, 15);	//Splitter for evaluating the model
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Creating PyramidDenseSIFT");
		DenseSIFT dsift = new DenseSIFT(5, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);										//Created a Pyramid Dense SIFT
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training Quantiser");
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), pdsift);			//Trains Pyramid Dense SIFT with K Means Clustering
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Done");
		
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);									//Creates a new instance of PHOW extractor below
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Wrapping Extractor");
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<? extends FeatureVector, FImage> wrappedExtractor = hkm.createWrappedExtractor(extractor); 		//Wraps extractor in a homogenous kernel map
		
		llann = new LibLinear(wrappedExtractor);																			//Create classifiers of the types to vote
		nbann = new NaiveBayes(wrappedExtractor);
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training LibLinearAnnotator");
		llann.train(splits.getTrainingDataset());																			//Train the classifiers
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training NaiveBayesAnnotator");
		nbann.train(splits.getTrainingDataset());
		
		llweight = llann.evaluate(splits.getTestDataset());								//Evaluate the classifiers and make their vote weighting the accuracy
		nbweight = nbann.evaluate(splits.getTestDataset());
	}
	
	public ArrayList<String> getVotes(FImage f) {
		ArrayList<String> votes = new ArrayList<String>();
		for(int i=0; i<llweight*100;i++) {												//Add a vote for each classifier times each of their weightings in %
			votes.add(llann.getVote(f));
		}
		for(int i=0;i<nbweight*100;i++) {
			votes.add(nbann.getVote(f));
		}
		
		return votes;
	}
	
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, PyramidDenseSIFT<FImage> pdsift)
	{
	    List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

	    for (FImage rec : groupedDataset) {
	        FImage img = rec.getImage();

	        pdsift.analyseImage(img);
	        allkeys.add(pdsift.getByteKeypoints(0.005f));
	    }
	    
	    int siftFeatures = 100;//10000
	    if (allkeys.size() > siftFeatures)
	        allkeys = allkeys.subList(0, siftFeatures);

	    ByteKMeans km = ByteKMeans.createKDTreeEnsemble(3);//300
	    DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
	    ByteCentroidsResult result = km.cluster(datasource);

	    return result.defaultHardAssigner();
	}
	
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
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
