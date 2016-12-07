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
	GIST gist;
	
	Run3() {
		try {
			trainingImages = new VFSGroupDataset<FImage>("C:/Users/brad/OpenIMAJ_CW3/training", ImageUtilities.FIMAGE_READER);
			testData = new VFSListDataset<FImage>("C:/Users/brad/OpenIMAJ_CW3/testing", ImageUtilities.FIMAGE_READER);
			out = new PrintWriter("run3.txt");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		Run3 r = new Run3();
		r.classifyImages();
	}

	public void classifyImages() {
		 phow = new PHOW(trainingImages);
		 gist = new GIST(trainingImages);
		
		
		FileObject[] files = testData.getFileObjects();
		String result;
		for(int i=0; i<testData.size(); i++) {
			result = files[i].getName().getBaseName()+" "+collateVotes(testData.get(i));
			out.println(result);
			System.out.println(result);
		}
		out.close();
	}
	
	
	private String collateVotes(FImage f) {
		HashMap<String, Integer> votes = new HashMap<String, Integer>();
		String curVote = phow.getVote(f);
		try {
			votes.put(curVote, votes.get(curVote)+1);
		} catch(Exception e) {
			votes.put(curVote, 1);
		}
		
		int max = 0;
		String maxS = "";
		for(String c: votes.keySet()) {
			if(votes.get(c)>max) {
				maxS = c;
				max = votes.get(c);
			}
		}
		
		return maxS;
	}
	
	
	
}
