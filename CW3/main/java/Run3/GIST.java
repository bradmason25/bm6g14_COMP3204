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
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.FloatDSIFTKeypoint;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import Run3.PHOW.PHOWExtractor;

public class GIST {
	VFSGroupDataset<FImage> trainingImages;
	LiblinearAnnotator<FImage, String> ann;
	
	GIST(VFSGroupDataset<FImage> trainingImages) {
		this.trainingImages = trainingImages;
		
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(trainingImages, 15, 0, 15);
		
		Gist gsift = new Gist();
		
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), gsift);
		
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(gsift, assigner);
		
		HomogeneousKernelMap hkm = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		FeatureExtractor<? extends FeatureVector, FImage> wrappedExtractor = hkm.createWrappedExtractor(extractor);
	}
	
	public String getVote(FImage f) {
		return ann.classify(f).getPredictedClasses().iterator().next();
	}
	
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, Gist<FImage> gsift)
	{
	    List<LocalFeatureList<FloatDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<FloatDSIFTKeypoint>>();

	    for (FImage rec : groupedDataset) {
	        FImage img = rec.getImage();

	        gsift.analyseImage(img);
	        allkeys.add(gsift.
	    }

	    if (allkeys.size() > 10000)
	        allkeys = allkeys.subList(0, 10000);

	    ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
	    DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
	    ByteCentroidsResult result = km.cluster(datasource);

	    return result.defaultHardAssigner();
	}
}
