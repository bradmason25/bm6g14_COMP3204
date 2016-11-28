package uk.ac.soton.ecs.bm6g14.ch13;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.model.EigenImages;

public class Tutorial13_1_3 {
	public static void main(String[] args) {
		try {
			
			VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>("zip:http://datasets.openimaj.org/att_faces.zip", ImageUtilities.FIMAGE_READER);
			
			int nTraining = 3;
			int nTesting = 5;
			GroupedRandomSplitter<String, FImage> splits = 
			    new GroupedRandomSplitter<String, FImage>(dataset, nTraining, 0, nTesting);
			GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
			GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
			
			List<FImage> basisImages = DatasetAdaptors.asList(training);
			int nEigenvectors = 100;
			EigenImages eigen = new EigenImages(nEigenvectors);
			eigen.train(basisImages);
			
			List<FImage> eigenFaces = new ArrayList<FImage>();
			for (int i = 0; i < 12; i++) {
			    eigenFaces.add(eigen.visualisePC(i));
			}
			DisplayUtilities.display("EigenFaces", eigenFaces);
			
			Map<String, DoubleFV[]> features = new HashMap<String, DoubleFV[]>();
			for (final String person : training.getGroups()) {
			    final DoubleFV[] fvs = new DoubleFV[nTraining];

			    for (int i = 0; i < nTraining; i++) {
			        final FImage face = training.get(person).get(i);
			        fvs[i] = eigen.extractFeature(face);
			    }
			    features.put(person, fvs);
			}
			
			double correct = 0, incorrect = 0;
			for (String truePerson : testing.getGroups()) {
			    for (FImage face : testing.get(truePerson)) {
			        DoubleFV testFeature = eigen.extractFeature(face);
			
			        String bestPerson = null;
			        double minDistance = Double.MAX_VALUE;
			        for (final String person : features.keySet()) {
			            for (final DoubleFV fv : features.get(person)) {
			                double distance = fv.compare(testFeature, DoubleFVComparison.EUCLIDEAN);
			
			                if (distance < minDistance) {
			                    minDistance = distance;
			                    bestPerson = person;
			                }
			            }
			        }
			        
			        //Here I compare the closest distance to a threshold
			        if(!(minDistance<12)) {
			        	bestPerson = null;
			        }
			
			        System.out.println("Actual: " + truePerson + "\tguess: " + bestPerson);
			
			        if (truePerson.equals(bestPerson))
			            correct++;
			        else
			            incorrect++;
			    }
			    //The use of a threshold may not be obvious in this situation but can be very useful elsewhere.
			    //Once the threshold has been tuned to the correct value it will return the same accuracy as the
			    //algorithm without the threshold, it's purpose being that rather than returning a wrong value in
			    //the case of an inaccuracy it will instead return null which can be ignored.
			    //Hence it helps because it knows when it gets the answer wrong.
			}
			
			System.out.println("Accuracy: " + (correct / (correct + incorrect)));
		} catch(Exception e) { e.printStackTrace(); }
	}
}
