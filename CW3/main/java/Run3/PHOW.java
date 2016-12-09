package Run3;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.Classifier;
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
//import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator.Mode;
//import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;

import de.bwaldvogel.liblinear.SolverType;

public class PHOW {
	VFSGroupDataset<FImage> trainingImages;
	LiblinearAnnotator<FImage, String> llann;
	NaiveBayesAnnotator<FImage, String> nbann;
	long startTime;

	PHOW(VFSGroupDataset<FImage> trainingImages, long startTime) {
		this.startTime = startTime;
		this.trainingImages = trainingImages;
		
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 15, 0, 15);
		
		DenseSIFT dsift = new DenseSIFT(5, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Created PyramidDenseSIFT");
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training Quantiser");
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), pdsift);
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Done");
		
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<? extends FeatureVector, FImage> wrappedExtractor = hkm.createWrappedExtractor(extractor);
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Wrapped Extractor");
		
		//The use of the wrapped extractor took slightly longer but did return higher accuracy compared to the previous extractor
		
		
		//ann = new LiblinearAnnotator<FImage, String>(wrappedExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		llann = new LiblinearAnnotator<FImage, String>(extractor, org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		nbann = new NaiveBayesAnnotator<FImage, String>(wrappedExtractor, org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator.Mode.ALL);
		
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training LibLinearAnnotator");
		llann.train(splits.getTrainingDataset());
		System.out.println((System.currentTimeMillis()-startTime)+"ms - Training NaiveBayesAnnotator");
		nbann.train(splits.getTrainingDataset());
	}
	
	public String getLinearVote(FImage f) {
		return llann.classify(f).getPredictedClasses().iterator().next();
	}
	public String getNaiveBayesVote(FImage f) {
		return nbann.classify(f).getPredictedClasses().iterator().next();
	}
	
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, PyramidDenseSIFT<FImage> pdsift)
	{
	    List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

	    for (FImage rec : groupedDataset) {
	        FImage img = rec.getImage();

	        pdsift.analyseImage(img);
	        allkeys.add(pdsift.getByteKeypoints(0.005f));
	    }

	    if (allkeys.size() > 10000)
	        allkeys = allkeys.subList(0, 10000);

	    ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
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
