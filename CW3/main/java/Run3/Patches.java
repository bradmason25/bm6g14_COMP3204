package Run3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

public class Patches extends Classifier{
	LiblinearAnnotator<FImage, String> annotator;
	
	//Implementation of the classifier class with the DSPP annotator implementation
	
	Patches() {
		annotator = null;
		name = "Patches";
	}
	
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> dataset) {
		annotator = new DenselySampledPixelPatches().getAnnotator(dataset);															//Train the classifier
	}
	
	
}
