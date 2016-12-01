package uk.ac.soton.ecs.bm6g14.ch6;

import java.util.Map.Entry;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class Tutorial6_1_1 {
	public static void main(String[] args) {
		try {
			
			VFSListDataset<FImage> images = new VFSListDataset<FImage>("/home/brad/Pictures/sample", ImageUtilities.FIMAGE_READER);
			
			VFSGroupDataset<FImage> groupedFaces = 
					new VFSGroupDataset<FImage>( "zip:http://datasets.openimaj.org/att_faces.zip", ImageUtilities.FIMAGE_READER);
			//I create an array of images to collect
			FImage[] faces = new FImage[groupedFaces.entrySet().size()];
			int i = 0;
			for (final Entry<String, VFSListDataset<FImage>> entry : groupedFaces.entrySet()) {
				//This loops through each directory and for each directory I add a random face to the array
				faces[i] = entry.getValue().getRandomInstance();
				i++;
			}
			DisplayUtilities.display("Random Faces", faces);
			System.out.println(faces.length);
			
			
		} catch(Exception e) { e.printStackTrace(); }
	}
}
