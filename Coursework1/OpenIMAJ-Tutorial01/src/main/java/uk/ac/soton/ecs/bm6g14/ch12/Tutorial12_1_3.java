package uk.ac.soton.ecs.bm6g14.ch12;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class Tutorial12_1_3 {
	public static void main(String[] args) {
		try {
			long startTime = System.currentTimeMillis();
			
			GroupedDataset<String, VFSListDataset<Record<FImage>>, Record<FImage>> allData = Caltech101.getData(ImageUtilities.FIMAGE_READER);
			
			GroupedDataset<String, ListDataset<Record<FImage>>, Record<FImage>> data = GroupSampler.sample(allData, 101, false);
			
			GroupedRandomSplitter<String, Record<FImage>> splits = new GroupedRandomSplitter<String, Record<FImage>>(data, 15, 0, 15);
			
			DenseSIFT dsift = new DenseSIFT(3, 7);
			PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4,6,8,10);
			
			HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 2), pdsift);
			
			IOUtils.writeToFile(assigner, new File("assigner.txt"));
			
			FeatureExtractor<DoubleFV, Record<FImage>> extractor = new PHOWExtractor(pdsift, assigner);
			DiskCachingFeatureExtractor<DoubleFV, Record<FImage>> cache = new DiskCachingFeatureExtractor<DoubleFV, Record<FImage>>(new File("cache"), extractor);
			
			
			LiblinearAnnotator<Record<FImage>, String> ann = new LiblinearAnnotator<Record<FImage>, String>(cache, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
			ann.train(splits.getTrainingDataset());
			
			ClassificationEvaluator<CMResult<String>, String, Record<FImage>> eval = new ClassificationEvaluator<CMResult<String>, String, Record<FImage>>(ann, splits.getTestDataset(), new CMAnalyser<Record<FImage>, String>(CMAnalyser.Strategy.SINGLE));
						
			Map<Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
			CMResult<String> result = eval.analyse(guesses);
			
			System.out.println(result.getDetailReport());
			
			long endTime = System.currentTimeMillis();
			System.out.println("The program ran in: "+ (endTime-startTime) + "milliseconds");

		} catch(Exception e) {}
	}
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<Record<FImage>> sample, PyramidDenseSIFT<FImage> pdsift)
	{
	    List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

	    for (Record<FImage> rec : sample) {
	        FImage img = rec.getImage();

	        pdsift.analyseImage(img);
	        allkeys.add(pdsift.getByteKeypoints(0.005f));
	    }

	    if (allkeys.size() > 10000)
	        allkeys = allkeys.subList(0, 10000);

	    ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600);
	    DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
	    ByteCentroidsResult result = km.cluster(datasource);

	    return result.defaultHardAssigner();
	}
	
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, Record<FImage>> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(Record<FImage> object) {
	        FImage image = object.getImage();
	        pdsift.analyseImage(image);

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 4);

	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}
}
